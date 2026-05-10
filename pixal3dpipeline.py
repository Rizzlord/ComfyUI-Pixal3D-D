"""
Pixal3D Pipeline
"""

import os
import shutil
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List, Tuple
from PIL import Image
from tqdm import tqdm
import trimesh
import json
from omegaconf import OmegaConf
import torchvision.transforms.functional as TF
from torchvision import transforms
import pixal3d
import sys
from pixal3d.modules import sparse as sp
from pixal3d.utils import postprocess_mesh, normalize_mesh, mesh2index, instantiate_from_config
from pixal3d.utils.sparse import sort_block


import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn


def _meshlib_postprocess(mesh, target_face_count=200000, remove_floaters=True):
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    mr_mesh = mn.meshFromFacesVerts(faces, verts)

    if remove_floaters:
        comps = mm.getAllComponents(mr_mesh)
        if len(comps) > 1:
            largest = max(comps, key=lambda c: c.count())
            to_del = mm.FaceBitSet(largest)
            to_del.flip()
            mr_mesh.deleteFaces(to_del)
            mr_mesh.invalidateCaches()
            print(f"[postprocess] removed {len(comps)-1} floater components")

    if target_face_count > 0 and mr_mesh.topology.numValidFaces() > target_face_count:
        settings = mm.DecimateSettings()
        settings.maxDeletedFaces = mr_mesh.topology.numValidFaces() - target_face_count
        settings.packMesh = True
        result = mm.decimateMesh(mr_mesh, settings)
        print(f"[postprocess] decimated: {result.vertsDeleted} verts, {result.facesDeleted} faces removed")

    out_verts = mn.getNumpyVerts(mr_mesh)
    out_faces = mn.getNumpyFaces(mr_mesh.topology)
    print(f"[postprocess] final: {len(out_verts)} verts, {len(out_faces)} faces")
    return trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=False)


def preprocess_image(image, resolution=518, padding=20, bg="white"):
    """
    Preprocess image for model input. Supports str path, PIL Image, or numpy array.
    Returns tensor [4, H, W] for model input.
    """
    # Handle different input types
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    if img.mode == 'RGB':
        resized = img.resize((resolution, resolution), Image.Resampling.BICUBIC)
        img_np = np.array(resized).astype(np.float32) / 255.0
        mask = np.ones((resolution, resolution, 1), dtype=np.float32)
        img_rgba = np.concatenate([img_np, mask], axis=-1)
    else:

        img = img.convert('RGBA')
        bbox = img.getbbox()
        
        if bbox is None:

            bg_val = 255 if bg == 'white' else (128 if bg == 'gray' else np.random.randint(0, 256))
            img_np = np.ones((resolution, resolution, 3), dtype=np.float32) * (bg_val / 255.0)
            mask = np.ones((resolution, resolution, 1), dtype=np.float32)
            img_rgba = np.concatenate([img_np, mask], axis=-1)
        else:
 
            cropped = img.crop(bbox)
            
       
            if bg == 'white':
                bg_color = (255, 255, 255, 255)
            elif bg == 'gray':
                bg_color = (128, 128, 128, 255)
            elif bg == 'random':
                bg_color = tuple(np.random.randint(0, 256, size=3).tolist()) + (255,)
            else:
                bg_color = (255, 255, 255, 255)
            

            bg_layer = Image.new('RGBA', cropped.size, bg_color)
            cropped_rgb = Image.alpha_composite(bg_layer, cropped).convert('RGB')
            
   
            target_size = resolution - padding * 2
            w, h = cropped_rgb.size
            scale = min(target_size / w, target_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
     
            resized = cropped_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
       
            result = Image.new('RGB', (resolution, resolution), bg_color[:3])
            offset_x = (resolution - new_w) // 2
            offset_y = (resolution - new_h) // 2
            result.paste(resized, (offset_x, offset_y))
            
            img_np = np.array(result).astype(np.float32) / 255.0
            mask = np.ones((resolution, resolution, 1), dtype=np.float32)
            img_rgba = np.concatenate([img_np, mask], axis=-1)
    

    tensor = torch.from_numpy(img_rgba).permute(2, 0, 1)
    return tensor


def compute_f_pixels(camera_angle_x, resolution):
    """
    Compute focal length in pixels
    """
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))  # mm
    f_pixels = focal_length * resolution / 32.0  # pixels
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    """
    Derive distance from FOV using analytical relationship.
    Returns distance derived from X and Y axes and focal length in pixels.
    """
    gp = grid_point.to(torch.float32)
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())

    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)

    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0) 


    eps = 1e-8
    if abs(x_ndc) < eps:
        raise ValueError("x_ndc too small to stably derive distance from X coordinate")
    if abs(y_ndc) < eps:
        raise ValueError("y_ndc too small to stably derive distance from Y coordinate")


    distance_x = f_pixels * xw / x_ndc - yw


    distance_y = f_pixels * zw / y_ndc - yw

    return {
        "distance_from_x": float(distance_x),
        "distance_from_y": float(distance_y),
        "f_pixels": float(f_pixels),
    }


# ==================== Pixal3D Pipeline ====================

class Pixal3DPipeline:
    """
    Pixal3D unified inference pipeline
    
    Self-contained pipeline integrating Dense and Sparse (512/1024) three-stage inference
    """
    
    def __init__(
        self,
        dense_visual_condition,
        dense_denoiser_model,
        dense_scheduler,
        sparse_512_visual_condition,
        sparse_512_denoiser_model,
        sparse_512_scheduler,
        sparse_1024_visual_condition,
        sparse_1024_denoiser_model,
        sparse_1024_scheduler,
        dense_vae,
        sparse_vae_512,
        sparse_vae_1024,
        dense_dtype: torch.dtype = torch.float16,
        sparse_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize Pixal3D Pipeline
        
        Args:
            dense_visual_condition: Dense visual condition encoder
            dense_denoiser_model: Dense denoising model
            dense_scheduler: Dense scheduler
            sparse_512_visual_condition: Sparse 512 visual condition encoder
            sparse_512_denoiser_model: Sparse 512 denoising model
            sparse_512_scheduler: Sparse 512 scheduler
            sparse_1024_visual_condition: Sparse 1024 visual condition encoder
            sparse_1024_denoiser_model: Sparse 1024 denoising model
            sparse_1024_scheduler: Sparse 1024 scheduler
            dense_vae: Dense VAE model
            sparse_vae_512: Sparse VAE 512 model
            sparse_vae_1024: Sparse VAE 1024 model
            dense_dtype: Dense model dtype (default fp16)
            sparse_dtype: Sparse model dtype (default bf16)
        """
        self.dense_visual_condition = dense_visual_condition
        self.dense_denoiser_model = dense_denoiser_model
        self.dense_scheduler = dense_scheduler
        
        self.sparse_512_visual_condition = sparse_512_visual_condition
        self.sparse_512_denoiser_model = sparse_512_denoiser_model
        self.sparse_512_scheduler = sparse_512_scheduler
        
        self.sparse_1024_visual_condition = sparse_1024_visual_condition
        self.sparse_1024_denoiser_model = sparse_1024_denoiser_model
        self.sparse_1024_scheduler = sparse_1024_scheduler
        
        self.dense_vae = dense_vae
        self.sparse_vae_512 = sparse_vae_512
        self.sparse_vae_1024 = sparse_vae_1024
        
        self.dense_dtype = dense_dtype
        self.sparse_dtype = sparse_dtype
        self.keep_model_loaded = True
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._active_stage = None
        
        # Set evaluation mode
        self._set_eval_mode()
        
    def _set_eval_mode(self):
        """Set all models to evaluation mode"""
        self.dense_visual_condition.eval()
        self.dense_denoiser_model.eval()
        self.sparse_512_visual_condition.eval()
        self.sparse_512_denoiser_model.eval()
        self.sparse_1024_visual_condition.eval()
        self.sparse_1024_denoiser_model.eval()
        self.dense_vae.eval()
        self.sparse_vae_512.eval()
        self.sparse_vae_1024.eval()
        
    def to(self, device):
        """Move all models to specified device"""
        self.device = device
        self.offload_models = False
        self.dense_visual_condition.to(device)
        self.dense_denoiser_model.to(device)
        self.sparse_512_visual_condition.to(device)
        self.sparse_512_denoiser_model.to(device)
        self.sparse_1024_visual_condition.to(device)
        self.sparse_1024_denoiser_model.to(device)
        self.dense_vae.to(device)
        self.sparse_vae_512.to(device)
        self.sparse_vae_1024.to(device)
        return self

    def _stage_module_names(self):
        return {
            "dense": [
                "dense_visual_condition",
                "dense_denoiser_model",
                "dense_vae",
            ],
            "sparse512_cond": [
                "sparse_512_visual_condition",
            ],
            "sparse512_dit": [
                "sparse_512_denoiser_model",
            ],
            "sparse512_vae": [
                "sparse_vae_512",
            ],
            "sparse1024_cond": [
                "sparse_1024_visual_condition",
            ],
            "sparse1024_dit": [
                "sparse_1024_denoiser_model",
            ],
            "sparse1024_vae": [
                "sparse_vae_1024",
            ],
        }

    def _flush_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_stage(self, stage: str, device: str):
        for attr_name in self._stage_module_names().get(stage, []):
            module = getattr(self, attr_name, None)
            if module is not None:
                module.to(device)

    def _move_all_stages(self, device: str):
        moved = set()
        for stage, attr_names in self._stage_module_names().items():
            for attr_name in attr_names:
                if attr_name in moved:
                    continue
                module = getattr(self, attr_name, None)
                if module is not None:
                    module.to(device)
                moved.add(attr_name)
        if str(device).startswith("cpu"):
            self._flush_cuda()

    def enable_model_cpu_offload(self, device: str = "cuda", offload_device: str = "cpu"):
        """Keep inactive stages on CPU and move only the current stage to GPU."""
        self.device = device
        self.offload_device = offload_device
        self.offload_models = True
        self._active_stage = None
        self._move_all_stages(offload_device)
        return self

    def disable_model_cpu_offload(self, device: str = "cuda"):
        self.offload_models = False
        self._active_stage = None
        return self.to(device)

    def _ensure_stage(self, stage: str):
        if not self.offload_models:
            return
        if self._active_stage == stage:
            return
        for other_stage in self._stage_module_names():
            if other_stage != stage:
                self._move_stage(other_stage, self.offload_device)
        self._flush_cuda()
        self._move_stage(stage, self.device)
        self._active_stage = stage

    def _offload_stage(self, stage: str):
        if not self.offload_models:
            return
        self._move_stage(stage, self.offload_device)
        if self._active_stage == stage:
            self._active_stage = None
        self._flush_cuda()

    def unload(self):
        """Delete all models and clear VRAM/RAM"""
        self.dense_vae = None
        self.dense_denoiser_model = None
        self.dense_visual_condition = None
        self.sparse_vae_512 = None
        self.sparse_512_denoiser_model = None
        self.sparse_512_visual_condition = None
        self.sparse_vae_1024 = None
        self.sparse_1024_denoiser_model = None
        self.sparse_1024_visual_condition = None
        
        gc.collect()
        torch.cuda.empty_cache()

    def offload_all_models(self):
        if self.offload_models:
            self._move_all_stages(self.offload_device)
            self._active_stage = None
    
    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str = "./ckpt",
        repo_id: str = None,
        dense_dtype: torch.dtype = torch.float16,
        sparse_dtype: torch.dtype = torch.float16,
        cache_dir: str = None,
        load_device: str = "cpu",
    ):
        """
        Create Pixal3D Pipeline from local directory or HuggingFace Hub.
        
        Args:
            ckpt_dir: Local directory containing converted checkpoints (used when repo_id is None)
            repo_id: HuggingFace repo ID (e.g., "TencentARC/Pixal3D-D"). If provided, download from HF Hub.
            dense_dtype: Data type for dense stage
            sparse_dtype: Data type for sparse stages
            cache_dir: Cache directory for downloaded models (default: ~/.cache/huggingface/hub)
        
        Usage:
            # Load from local directory
            pipeline = Pixal3DPipeline.from_ckpt("./ckpt")
            
            # Load from HuggingFace Hub
            pipeline = Pixal3DPipeline.from_ckpt(repo_id="TencentARC/Pixal3D-D")
        """
        import json
        import importlib
        from safetensors.torch import load_file
        
        # Determine source
        if repo_id is not None:
            # Load from HuggingFace Hub
            from huggingface_hub import hf_hub_download, snapshot_download
            use_hf_hub = True
            print(f"Loading models from HuggingFace Hub: {repo_id}")
        else:
            # Load from local directory
            use_hf_hub = False
            print(f"Loading models from local directory: {ckpt_dir}")
        
        def get_component_path(stage: str, component: str) -> str:
            """Get path to component directory."""
            if use_hf_hub:
                return f"{stage}/{component}"
            else:
                return os.path.join(ckpt_dir, stage, component)
        
        def load_config_hf(repo_id: str, subfolder: str, cache_dir: str = None):
            """Load config.json from HuggingFace Hub."""
            config_path = hf_hub_download(
                repo_id=repo_id,
                subfolder=subfolder,
                filename="config.json",
                cache_dir=cache_dir,
                repo_type="model"
            )
            with open(config_path, 'r') as f:
                return json.load(f)
        
        def load_config_local(component_dir: str):
            """Load config.json from local directory."""
            config_path = os.path.join(component_dir, "config.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        
        def load_config(stage: str, component: str):
            """Load config.json from appropriate source."""
            subfolder = f"{stage}/{component}"
            if use_hf_hub:
                return load_config_hf(repo_id, subfolder, cache_dir)
            else:
                return load_config_local(os.path.join(ckpt_dir, stage, component))
        
        def load_model_hf(repo_id: str, subfolder: str, device="cuda", cache_dir: str = None):
            """Load model from HuggingFace Hub."""
            config = load_config_hf(repo_id, subfolder, cache_dir)
            model_class_path = config["model_class"]
            
            # Import class
            module_path, class_name = model_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            # Build kwargs
            if config.get("model_type") == "conditioner":
                kwargs = config.get("config", {})
            else:
                exclude_keys = {"model_type", "model_class", "scheduler_class", "scheduler_config", "config"}
                kwargs = {k: v for k, v in config.items() if k not in exclude_keys}
            
            # Create model
            if hasattr(model_class, 'Config'):
                model = model_class(cfg=kwargs)
            else:
                model = model_class(**kwargs)
            
            # Load weights if exists (check config first)
            if config.get("model_type") not in ["scheduler", "conditioner"]:
                safetensors_path = hf_hub_download(
                    repo_id=repo_id,
                    subfolder=subfolder,
                    filename="model.safetensors",
                    cache_dir=cache_dir,
                    repo_type="model"
                )
                state_dict = load_file(safetensors_path, device=device)
                model.load_state_dict(state_dict, strict=True)
            
            return model
        
        def load_model_local(component_dir: str, device="cuda"):
            """Load model from local directory."""
            config = load_config_local(component_dir)
            model_class_path = config["model_class"]
            
            # Import class
            module_path, class_name = model_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            # Build kwargs
            if config.get("model_type") == "conditioner":
                kwargs = config.get("config", {})
            else:
                exclude_keys = {"model_type", "model_class", "scheduler_class", "scheduler_config", "config"}
                kwargs = {k: v for k, v in config.items() if k not in exclude_keys}
            
            # Create model
            if hasattr(model_class, 'Config'):
                model = model_class(cfg=kwargs)
            else:
                model = model_class(**kwargs)
            
            # Load weights if exists
            safetensors_path = os.path.join(component_dir, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path, device=device)
                model.load_state_dict(state_dict, strict=True)
            
            return model
        
        def load_model(stage: str, component: str, device="cuda"):
            """Load model from appropriate source."""
            subfolder = f"{stage}/{component}"
            if use_hf_hub:
                return load_model_hf(repo_id, subfolder, device, cache_dir)
            else:
                return load_model_local(os.path.join(ckpt_dir, stage, component), device)
        
        def load_scheduler(stage: str):
            """Load scheduler from appropriate source."""
            config = load_config(stage, "scheduler")
            scheduler_class_path = config["scheduler_class"]
            scheduler_config = config.get("scheduler_config", {})
            
            # Import class
            module_path, class_name = scheduler_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            scheduler_class = getattr(module, class_name)
            
            return scheduler_class(**scheduler_config)
        
        def load_conditioner(stage: str):
            config = load_config(stage, "conditioner")
            conditioner_class_path = config["model_class"]
            module_path, class_name = conditioner_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            conditioner_class = getattr(module, class_name)
            visual_condition = conditioner_class(cfg=config.get("config", {}))
            visual_condition.to(load_device)
            visual_condition.requires_grad_(False)
            return visual_condition
        
        dense_denoiser_model = load_model("dense", "dit", device=load_device)
        dense_vae = load_model("dense", "vae", device=load_device)
        dense_vae.eval()
        dense_scheduler = load_scheduler("dense")
        dense_visual_condition = load_conditioner("dense")
        
        sparse_512_denoiser_model = load_model("sparse512", "dit", device=load_device)
        sparse_vae_512 = load_model("sparse512", "vae", device=load_device)
        sparse_vae_512.eval()
        sparse_512_scheduler = load_scheduler("sparse512")
        sparse_512_visual_condition = load_conditioner("sparse512")
        
        sparse_1024_denoiser_model = load_model("sparse1024", "dit", device=load_device)
        sparse_vae_1024 = load_model("sparse1024", "vae", device=load_device)
        sparse_vae_1024.eval()
        sparse_1024_scheduler = load_scheduler("sparse1024")
        sparse_1024_visual_condition = load_conditioner("sparse1024")
        
        print("All models loaded successfully!")
        
        return cls(
            dense_visual_condition=dense_visual_condition,
            dense_denoiser_model=dense_denoiser_model,
            dense_scheduler=dense_scheduler,
            sparse_512_visual_condition=sparse_512_visual_condition,
            sparse_512_denoiser_model=sparse_512_denoiser_model,
            sparse_512_scheduler=sparse_512_scheduler,
            sparse_1024_visual_condition=sparse_1024_visual_condition,
            sparse_1024_denoiser_model=sparse_1024_denoiser_model,
            sparse_1024_scheduler=sparse_1024_scheduler,
            dense_vae=dense_vae,
            sparse_vae_512=sparse_vae_512,
            sparse_vae_1024=sparse_vae_1024,
            dense_dtype=dense_dtype,
            sparse_dtype=sparse_dtype,
        )
    
    # ==================== Image Encoding ====================
    
    def encode_image_dense(self, image, camera_angle_x, distance, mesh_scale):

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dense_dtype):
                cond_global, cond_proj = self.dense_visual_condition(
                    image[:, :3],
                    camera_angle_x=camera_angle_x,
                    distance=distance,
                    mesh_scale=mesh_scale,
                )
        

        uncond_global = torch.zeros_like(cond_global)
        uncond_proj = torch.zeros_like(cond_proj)
        
        return (cond_global, cond_proj), (uncond_global, uncond_proj)
    
    def encode_image_sparse(self, image, camera_angle_x, distance, mesh_scale, coords, visual_condition):

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.sparse_dtype):
                cond_global, cond_sparse = visual_condition(
                    image[:, :3],
                    camera_angle_x=camera_angle_x,
                    distance=distance,
                    mesh_scale=mesh_scale,
                )
            
  
            bs = cond_sparse.shape[0]
            res = visual_condition.grid_resolution
            cond_sparse = cond_sparse.reshape(bs, res, res, res, -1)
            
            batch_indices = coords[:, 0].long()
            x_coords = coords[:, 1].long()
            y_coords = coords[:, 2].long()
            z_coords = coords[:, 3].long()
            
            cond_sparse = cond_sparse[batch_indices, x_coords, y_coords, z_coords]
            
 
            uncond_global = torch.zeros_like(cond_global)
            uncond_sparse = torch.zeros_like(cond_sparse)
            
     
            cond_sparse = sp.SparseTensor(cond_sparse, coords.int())
            uncond_sparse = sp.SparseTensor(uncond_sparse, coords.int())
        
        return (cond_global, cond_sparse), (uncond_global, uncond_sparse)

    def encode_image_sparse_cross_res(self, image, camera_angle_x, distance, mesh_scale,
                                       coords, visual_condition, target_grid_resolution):
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.sparse_dtype):
                cond_global, cond_sparse = visual_condition(
                    image[:, :3],
                    camera_angle_x=camera_angle_x,
                    distance=distance,
                    mesh_scale=mesh_scale,
                )

            bs = cond_sparse.shape[0]
            src_res = visual_condition.grid_resolution
            cond_sparse = cond_sparse.reshape(bs, src_res, src_res, src_res, -1)
            cond_volume = cond_sparse.permute(0, 4, 1, 2, 3).contiguous()
            del cond_sparse

            spatial = coords[:, 1:].float()
            spatial = spatial / (target_grid_resolution - 1) * 2.0 - 1.0
            spatial = spatial.flip(-1)
            grid = spatial.unsqueeze(0).unsqueeze(2).unsqueeze(2)

            sampled = F.grid_sample(
                cond_volume, grid, mode='bilinear',
                align_corners=True, padding_mode='border'
            )
            del cond_volume

            cond_sparse_out = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)
            cond_sparse_out = cond_sparse_out.squeeze(0)
            del sampled

            uncond_global = torch.zeros_like(cond_global)
            uncond_sparse = torch.zeros_like(cond_sparse_out)

            cond_sparse_out = sp.SparseTensor(cond_sparse_out, coords.int())
            uncond_sparse = sp.SparseTensor(uncond_sparse, coords.int())

        return (cond_global, cond_sparse_out), (uncond_global, uncond_sparse)

    
    @torch.no_grad()
    def infer_dense(self, image, camera_angle_x, distance, mesh_scale, num_steps, guidance_scale, seed):
        self._ensure_stage("dense")

        batch_size = image.shape[0]
        
        # Encode conditions
        do_cfg = guidance_scale > 0
        image = image.to(torch.float16)
        cond, uncond = self.encode_image_dense(image, camera_angle_x, distance, mesh_scale)
        
        # Initialize latents
        latent_shape = (batch_size, *self.dense_denoiser_model.dit_model.latent_shape)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        latents = torch.randn(latent_shape, device=self.device, dtype=cond[0].dtype, generator=generator)
        
        # Setup scheduler
        self.dense_scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.dense_scheduler.timesteps
        
        extra_step_kwargs = {'generator': generator} if generator is not None else {}
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Dense Sampling")):
            timestep_tensor = torch.tensor([t], dtype=latents.dtype, device=self.device)
            
            diffusion_inputs = {"x": latents, "t": timestep_tensor,"cond": cond,}
            
            with torch.cuda.amp.autocast(dtype=self.dense_dtype):
                noise_pred_cond = self.dense_denoiser_model(**diffusion_inputs).sample
                
                if do_cfg:
                    diffusion_inputs["cond"] = uncond
                    noise_pred_uncond = self.dense_denoiser_model(**diffusion_inputs).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond
            
            latents = self.dense_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        
        return latents
    
    @torch.no_grad()
    def infer_sparse(self, image, camera_angle_x, distance, mesh_scale, index, num_steps, guidance_scale, seed, 
                     visual_condition, denoiser_model, scheduler, cross_res_cond=None):
        if denoiser_model is self.sparse_512_denoiser_model:
            cond_stage = "sparse512_cond"
            dit_stage = "sparse512_dit"
        elif denoiser_model is self.sparse_1024_denoiser_model:
            cond_stage = "sparse1024_cond"
            dit_stage = "sparse1024_dit"
        else:
            cond_stage = None
            dit_stage = None

        do_cfg = guidance_scale > 0
        if cross_res_cond is not None:
            src_cond, src_cond_stage, target_grid_res = cross_res_cond
            self._ensure_stage(src_cond_stage)
            cond, uncond = self.encode_image_sparse_cross_res(
                image, camera_angle_x, distance, mesh_scale, index, src_cond, target_grid_res
            )
            self._offload_stage(src_cond_stage)
        else:
            if cond_stage:
                self._ensure_stage(cond_stage)
            batch_size = image.shape[0]
            cond, uncond = self.encode_image_sparse(image, camera_angle_x, distance, mesh_scale, index, visual_condition)
            if cond_stage:
                self._offload_stage(cond_stage)

        if dit_stage:
            self._ensure_stage(dit_stage)

        latent_shape = (index.shape[0], denoiser_model.dit_model.out_channels)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        latents = torch.randn(latent_shape, device=self.device, dtype=cond[0].dtype, generator=generator)

        scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = scheduler.timesteps

        extra_step_kwargs = {'generator': generator} if generator is not None else {}

        for i, t in enumerate(tqdm(timesteps, desc="Sparse Sampling")):
            timestep_tensor = torch.tensor([t], dtype=latents.dtype, device=self.device)

            x_input = sp.SparseTensor(latents, index.int())

            diffusion_inputs = {
                "x": x_input,
                "t": timestep_tensor,
                "cond": cond,
            }

            with torch.cuda.amp.autocast(dtype=self.sparse_dtype):
                noise_pred_cond = denoiser_model(**diffusion_inputs).sample
                noise_pred_cond = noise_pred_cond.feats

                if do_cfg:
                    diffusion_inputs["cond"] = uncond
                    noise_pred_uncond = denoiser_model(**diffusion_inputs).sample
                    noise_pred_uncond = noise_pred_uncond.feats
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        return sp.SparseTensor(latents, index.int())
    
    # ==================== Main Inference Interface ====================
    
    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Image.Image, np.ndarray],
        dense_steps: int = 50,
        dense_guidance_scale: float = 7.0,
        dense_seed: int = 0,
        sparse_512_steps: int = 30,
        sparse_512_guidance_scale: float = 7.0,
        sparse_1024_steps: int = 15,
        sparse_1024_guidance_scale: float = 7.0,
        sparse_seed: int = 0,
        dense_threshold: float = 0.1,
        mc_threshold: float = 0.2,

        extend_pixel: int = 20,
        camera_angle_x: float = 0.2,
        mesh_scale: float = 0.9,
        mode_1024: str = "refine",
        target_face_count: int = 200000,
        remove_floaters: bool = True,
    ):
        # Image preprocessing (always executed)
        image_tensor = preprocess_image(image, 518, padding=20).unsqueeze(0).to(self.device)
        
        # Compute camera distance 
        image_resolution = 518
        grid_points = torch.tensor([-1.0, 0, -1.0])
        grid_points = grid_points / mesh_scale / 2
        distance = distance_from_fov(
            camera_angle_x, grid_points, torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]), mesh_scale, image_resolution
        )["distance_from_x"]
        
        print(f"[Pixal3D] camera_angle_x: {camera_angle_x}, distance: {distance}")
        

        camera_angle_x_tensor = torch.tensor([camera_angle_x], device=self.device, dtype=torch.float32)
        distance_tensor = torch.tensor([distance], device=self.device, dtype=torch.float32)
        mesh_scale_tensor = torch.tensor([mesh_scale], device=self.device, dtype=torch.float32)
        

       
        
        # ============ Step 1: Dense Inference ============
        print(f"[Pixal3D] Step 1: Dense Inference...")
        dense_latents = self.infer_dense(
            image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor,
            dense_steps, dense_guidance_scale, dense_seed
        )
        
        with torch.autocast("cuda", dtype=torch.float16):
            decoded_index = self.dense_vae.decode_mesh(
                dense_latents, mc_threshold=dense_threshold, return_index=True
            )[0]
        
        del dense_latents
        decoded_index = sort_block(decoded_index, 8)
        print(f"[Pixal3D] decoded_index max: {decoded_index.max(0)}, min: {decoded_index.min(0)}, shape: {decoded_index.shape}")
        
        print(f"[Pixal3D] Step 2: Sparse 512 Inference...")
        sparse_latents_512 = self.infer_sparse(
            image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor, decoded_index,
            sparse_512_steps, sparse_512_guidance_scale, sparse_seed,
            self.sparse_512_visual_condition, self.sparse_512_denoiser_model, self.sparse_512_scheduler
        )
        
        self._offload_stage("sparse512_dit")
        self._ensure_stage("sparse512_vae")
        with torch.autocast("cuda", dtype=torch.float16):
            with torch.no_grad():
                decoded_meshs_512 = self.sparse_vae_512.decode_mesh(sparse_latents_512, voxel_resolution=512)
                mesh_512 = decoded_meshs_512[0]
        
        del decoded_index, sparse_latents_512, decoded_meshs_512
        self._offload_stage("sparse512_vae")
        gc.collect()
        torch.cuda.empty_cache()

        rotation_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        if mode_1024 == "skip":
            print(f"[Pixal3D] mode_1024=skip, returning 512 mesh directly.")
            with torch.autocast("cuda", dtype=torch.float16):
                with torch.no_grad():
                    mesh_v, mesh_f = postprocess_mesh(
                        mesh_512.vertices, mesh_512.faces,
                        simplify=True, verbose=True,
                    )
                    result = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f, process=False)
                    result.apply_scale(0.5 / mesh_scale)
                    result.apply_transform(rotation_matrix)
            del mesh_512
            torch.cuda.empty_cache()
            self.offload_all_models()
            return _meshlib_postprocess(result, target_face_count, remove_floaters)
        
        print(f"[Pixal3D] Step 3: Prepare 1024 latent index...")
        latent_index_1024 = mesh2index(mesh_512, size=1024, factor=8)
        del mesh_512
        gc.collect()
        torch.cuda.empty_cache()
        block_size_1024 = getattr(self.sparse_1024_denoiser_model.dit_model, 'selection_block_size', 8)
        latent_index_1024 = sort_block(latent_index_1024, block_size_1024)
        print(f"[Pixal3D] 1024 latent tokens: {len(latent_index_1024)}")
        print(f"[VRAM] before 1024 inference: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        print(f"[Pixal3D] Step 4: Sparse 1024 Inference (mode={mode_1024})...")
        cross_res = None
        if mode_1024 == "refine":
            cross_res = (self.sparse_512_visual_condition, "sparse512_cond", 128)

        sparse_latents_1024 = self.infer_sparse(
            image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor, latent_index_1024,
            sparse_1024_steps, sparse_1024_guidance_scale, sparse_seed,
            self.sparse_1024_visual_condition, self.sparse_1024_denoiser_model, self.sparse_1024_scheduler,
            cross_res_cond=cross_res
        )
        
        self._offload_stage("sparse1024_dit")
        self._ensure_stage("sparse1024_vae")
        with torch.autocast("cuda", dtype=torch.float16):
            with torch.no_grad():
                decoded_meshs_1024 = self.sparse_vae_1024.decode_mesh(
                    sparse_latents_1024, voxel_resolution=1024, mc_threshold=mc_threshold
                )
                
                mesh_v, mesh_f = postprocess_mesh(
                    decoded_meshs_1024[0].vertices, decoded_meshs_1024[0].faces,
                    simplify=True, verbose=True,
                )
                mesh_1024 = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f, process=False)
                mesh_1024.apply_scale(0.5 / mesh_scale)
                mesh_1024.apply_transform(rotation_matrix)
        
        del latent_index_1024, sparse_latents_1024, decoded_meshs_1024
        torch.cuda.empty_cache()
        self.offload_all_models()
        
        return _meshlib_postprocess(mesh_1024, target_face_count, remove_floaters)
    
    def infer_from_image(self, image_path: str, **kwargs):
        return self.infer(image=image_path, **kwargs)
