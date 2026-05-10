# Some parts of this file are adapted from the SparseDiT implementation
import os
from typing import Any, Dict, Optional, Union, Tuple, Literal
from dataclasses import dataclass
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging

import pixal3d
from pixal3d.utils.base import BaseModule
from huggingface_hub import hf_hub_download

# Import sparse operations

from ...modules import sparse as sp
from ...modules.utils import convert_module_to_f16, convert_module_to_f32
from ...modules.transformer import AbsolutePositionEmbedder
from ...modules.sparse.transformer.modulated import ModulatedSparseTransformerCrossBlock
SPARSE_AVAILABLE = True
# except ImportError:
    # print("Warning: sparse modules not found. Please ensure it's in your Python path.")
    # sp = None
    # convert_module_to_f16 = None
    # convert_module_to_f32 = None
    # AbsolutePositionEmbedder = None
    # ModulatedSparseTransformerCrossBlock = None
    # SPARSE_AVAILABLE = False

logger = logging.get_logger(__name__)


@dataclass
class SparseDiTModelOutput:
    sample: Any  # Can be torch.FloatTensor or sp.SparseTensor


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    Sparse Diffusion Transformer model for 3D shape generation.
    
    This model processes sparse 3D data using sparse attention mechanisms.
    """
    
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        resolution: int = 64,
        in_channels: int = 16,
        model_channels: int = 1024,
        cond_channels: int = 1024,
        out_channels: int = 16,
        num_blocks: int = 24,
        num_heads: int = 32,
        num_head_channels: int = 64,
        num_kv_heads: int = 2,
        compression_block_size: int = 4,
        selection_block_size: int = 8,
        topk: int = 32,
        compression_version: str = 'v2',
        mlp_ratio: float = 4.0,
        pe_mode: str = "ape",
        use_fp16: bool = True,
        use_checkpoint: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
        sparse_conditions: bool = True,
        factor: float = 1.0,
        window_size: int = 8,
        use_shift: bool = True,
        image_attn_mode:str='cross',
        load_ckpt:bool=True,
        version:Optional[str]='V10',
    ):
        super().__init__()
        
        if not SPARSE_AVAILABLE:
            raise ImportError("sparse modules not found.")
        
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads  or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self._dtype = torch.float16 if use_fp16 else torch.float32
        self.sparse_conditions = sparse_conditions
        self.factor = factor
        self.compression_block_size = compression_block_size
        self.selection_block_size = selection_block_size
        self.image_attn_mode = image_attn_mode

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(model_channels)
        
        # Shared modulation if enabled
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
        
        # Condition processing for sparse conditions
        if sparse_conditions:
            self.cond_proj = sp.SparseLinear(cond_channels, cond_channels)
            self.pos_embedder_cond = AbsolutePositionEmbedder(model_channels, in_channels=3)

        # Position embedding
        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        # Input projection
        self.input_layer = sp.SparseLinear(in_channels, model_channels)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                num_kv_heads=num_kv_heads,
                compression_block_size=compression_block_size,
                selection_block_size=selection_block_size,
                topk=topk,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                compression_version=compression_version,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                resolution=resolution,
                window_size=window_size,
                shift_window=window_size // 2 * (i % 2) if use_shift else window_size // 2,
                image_attn_mode = image_attn_mode,
            )
            for i in range(num_blocks)
        ])
        
        # Output projection
        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        # Initialize weights
        self.initialize_weights()
  

        self.gradient_checkpointing = False

        if use_fp16:
            print("Converting model to float16 ============================")
            self.convert_to_fp16()
        # else:
            # self.convert_to_fp32()
    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def convert_to_fp16(self) -> None:
        """Convert the model to float16."""
        self.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Convert the model to float32."""
        self.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        """Initialize model weights."""
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                # if hasattr(block, 'adaLN_modulation'):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        hidden_states: Any,  # sp.SparseTensor
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[Any] = None,  # torch.Tensor or sp.SparseTensor
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[SparseDiTModelOutput, Tuple]:
        """
        Forward pass of the SparseDiT model.
        
        Args:
            hidden_states: Input sparse tensor
            timestep: Timestep tensor
            encoder_hidden_states: Condition tensor (visual/text conditions)
            attention_kwargs: Additional attention arguments
            return_dict: Whether to return a dictionary
        """
        # breakpoint()
        # Process input
        assert attention_kwargs is None, "attention_kwargs not supported in SparseDiT"
        # breakpoint()
        h = self.input_layer(hidden_states).type(self._dtype)
        
        # Process timestep
        t_emb = self.t_embedder(timestep)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self._dtype)
        
        # Process conditions
        
        cond = encoder_hidden_states
        if self.image_attn_mode=='proj':
            global_cond,sparse_cond = cond
            
            if sparse_cond is not None:
                sparse_cond = sparse_cond.type(self._dtype)
                global_cond = global_cond.type(self._dtype)
                # breakpoint()
                if self.sparse_conditions and isinstance(sparse_cond, sp.SparseTensor):
                    # breakpoint()
                    sparse_cond = self.cond_proj(sparse_cond)
                    sparse_cond = sparse_cond + self.pos_embedder_cond(sparse_cond.coords[:, 1:]).type(self._dtype)
                cond = (global_cond,sparse_cond)
        else:
            if self.sparse_conditions:
                cond = self.cond_proj(cond)
                cond = cond + self.pos_embedder_cond(cond.coords[:, 1:]).type(self.dtype)

        # Add positional embeddings
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:], factor=self.factor).type(self._dtype)
        
        # Process through transformer blocks
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    h, t_emb, cond
                )
            else:
                h = block(h, t_emb, cond)
        
        # Final layer norm and output projection
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(hidden_states.dtype))
        
        if not return_dict:
            return (h,)
        
        return SparseDiTModelOutput(sample=h)


@pixal3d.register("sparse-dit-denoiser")
class SparseDiTDenoiser(BaseModule):
    """
    Sparse DiT Denoiser wrapper for pixal3d framework.
    """
    
    @dataclass 
    class Config(BaseModule.Config):
        # Model architecture
        resolution: int = 64
        in_channels: int = 16
        model_channels: int = 1024
        cond_channels: int = 1024
        out_channels: int = 16
        num_blocks: int = 24
        num_heads: int = 32
        num_kv_heads: int = 2
        compression_block_size: int = 4
        selection_block_size: int = 8
        topk: int = 32
        compression_version: str = 'v2'
        mlp_ratio: float = 4.0
        pe_mode: str = "ape"
        use_fp16: bool = True
        use_checkpoint: bool = True
        qk_rms_norm: bool = True
        qk_rms_norm_cross: bool = False
        sparse_conditions: bool = True
        factor: float = 1.0
        window_size: int = 8
        use_shift: bool = True
        
        # Condition settings
        use_visual_condition: bool = True
        visual_condition_dim: int = 1024
        use_caption_condition: bool = False
        caption_condition_dim: int = 1024
        use_label_condition: bool = False
        label_condition_dim: int = 1024
        
        # Training settings
        pretrained_model_name_or_path: Optional[str] = None

        image_attn_mode:Optional[str]='cross'
        load_ckpt:bool =True
        version:Optional[str]='V10'

    cfg: Config

    def configure(self) -> None:
        """Configure the SparseDiT model."""
        
        # Create the core SparseDiT model
        self.dit_model = SparseDiTModel(
            resolution=self.cfg.resolution,
            in_channels=self.cfg.in_channels,
            model_channels=self.cfg.model_channels,
            cond_channels=self.cfg.cond_channels,
            out_channels=self.cfg.out_channels,
            num_blocks=self.cfg.num_blocks,
            num_heads=self.cfg.num_heads,
            num_kv_heads=self.cfg.num_kv_heads,
            compression_block_size=self.cfg.compression_block_size,
            selection_block_size=self.cfg.selection_block_size,
            topk=self.cfg.topk,
            compression_version=self.cfg.compression_version,
            mlp_ratio=self.cfg.mlp_ratio,
            pe_mode=self.cfg.pe_mode,
            use_fp16=self.cfg.use_fp16,
            use_checkpoint=self.cfg.use_checkpoint,
            sparse_conditions=self.cfg.sparse_conditions,
            factor=self.cfg.factor,
            window_size=self.cfg.window_size,
            use_shift=self.cfg.use_shift,
            image_attn_mode=self.cfg.image_attn_mode,
            load_ckpt = self.cfg.load_ckpt,
            version=self.cfg.version,
        )
        
        # Condition projectors
        if self.cfg.use_visual_condition and self.cfg.visual_condition_dim != self.cfg.cond_channels:
            self.proj_visual_condition = nn.Sequential(
                nn.RMSNorm(self.cfg.visual_condition_dim),
                nn.Linear(self.cfg.visual_condition_dim, self.cfg.cond_channels),
            )
            
        if self.cfg.use_caption_condition and self.cfg.caption_condition_dim != self.cfg.cond_channels:
            self.proj_caption_condition = nn.Sequential(
                nn.RMSNorm(self.cfg.caption_condition_dim),
                nn.Linear(self.cfg.caption_condition_dim, self.cfg.cond_channels),
            )
            
        if self.cfg.use_label_condition and self.cfg.label_condition_dim != self.cfg.cond_channels:
            self.proj_label_condition = nn.Sequential(
                nn.RMSNorm(self.cfg.label_condition_dim),
                nn.Linear(self.cfg.label_condition_dim, self.cfg.cond_channels),
            )

        # Load pretrained weights if specified
        if self.cfg.pretrained_model_name_or_path:
            print(f"Loading pretrained SparseDiT model from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(
                self.cfg.pretrained_model_name_or_path,
                map_location="cpu",
                weights_only=True,
            )
            if "state_dict" in ckpt.keys():
                ckpt = ckpt["state_dict"]
            self.load_state_dict(ckpt, strict=True)

    def forward(
        self,
        x: Any,  # sp.SparseTensor
        t: torch.Tensor,
        cond: Optional[Any] = None,
    ):
        """
        Forward pass of the denoiser.
        
        Args:
            model_input: Input sparse tensor [SparseTensor with features]
            timestep: Timestep tensor [batch_size,]
            visual_condition: Visual condition tensor
            caption_condition: Caption condition tensor
            label_condition: Label condition tensor
            attention_kwargs: Additional attention arguments
            return_dict: Whether to return a dictionary
        """
        
      
        output = self.dit_model(
            hidden_states=x,
            timestep=t,
            encoder_hidden_states=cond,
        )
        
        return output


