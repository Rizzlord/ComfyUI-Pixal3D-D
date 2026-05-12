"""
Pixal3D 2-Stage Pipeline

Extended pipeline with MoGe FOV estimation and iterative mesh_scale optimization.
Compared to the base Pixal3DPipeline (fixed camera_angle_x=0.2, mesh_scale=0.9),
this pipeline:
  1. Uses MoGe model to estimate camera FOV from the input image
  2. Iteratively optimizes mesh_scale so decoded indices fit within the grid
  3. Optionally uses a separate dense_check model for mesh_scale optimization
"""

import os
import math
import gc
import torch
import numpy as np
from typing import Union
from PIL import Image
import trimesh

from pixal3d.utils import postprocess_mesh, mesh2index
from pixal3d.utils.sparse import sort_block

from .pixal3dpipeline import (
    Pixal3DPipeline,
    preprocess_image,
    compute_f_pixels,
    distance_from_fov,
    _meshlib_postprocess,
)


def load_moge_model(device: str = "cuda", model_name: str = "Ruicheng/moge-vitl"):
    """Load MoGe model for FOV estimation."""
    print(f"[MoGe] Loading model {model_name}...")
    from moge.model.v1 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name).to(device)
    moge_model.eval()
    print("[MoGe] Model loaded!")
    return moge_model


def get_camera_angle_x_from_moge(image_path: str, moge_model, device: str = "cuda") -> float:
    """
    Estimate camera_angle_x (horizontal FOV in radians) via MoGe inference.
    
    Args:
        image_path: Input image path (must be square)
        moge_model: MoGe model instance
        device: Inference device
    
    Returns:
        camera_angle_x in radians
    """
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    assert width == height, f"Image must be square, but got {width}x{height}"

    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)  # [3, H, W]

    with torch.no_grad():
        output = moge_model.infer(image_tensor.unsqueeze(0))

    intrinsics = output["intrinsics"].squeeze(0).cpu().numpy()  # [3, 3]
    fx = intrinsics[0, 0] * width

    camera_angle_x = 2 * math.atan(width / (2 * fx))
    print(f"[MoGe] fx={fx:.2f}, width={width}, camera_angle_x={camera_angle_x:.6f} rad ({math.degrees(camera_angle_x):.2f} deg)")

    return camera_angle_x


def compute_optimal_mesh_scale(
    decoded_index: torch.Tensor,
    original_mesh_scale: float,
    grid_resolution: int = 64,
    padding: int = 3,
) -> float:
    """
    Compute optimal mesh_scale so decoded indices fill the grid with target padding.

    Args:
        decoded_index: [N, 4] tensor, [:, 1:4] are xyz indices in 64^3 grid
        original_mesh_scale: Current mesh scale factor
        grid_resolution: Grid resolution (default 64)
        padding: Target boundary distance in voxels (default 3)

    Returns:
        optimal_mesh_scale
    """
    xyz_index = decoded_index[:, 1:4].float()  # [N, 3]

    center = (grid_resolution - 1) / 2.0  # 31.5
    offset = xyz_index - center
    max_abs_offset = offset.abs().max().item()
    target_max_offset = center - padding  # 31.5 - 3 = 28.5

    if max_abs_offset > 0:
        scale_factor = target_max_offset / max_abs_offset
    else:
        scale_factor = 1.0

    optimal_mesh_scale = original_mesh_scale * scale_factor
    print(f"[compute_optimal_mesh_scale] max_abs_offset={max_abs_offset:.4f}, "
          f"scale_factor={scale_factor:.4f}, optimal_mesh_scale={optimal_mesh_scale:.6f}")

    return optimal_mesh_scale


class Pixal3DPipeline2Stage(Pixal3DPipeline):
    """
    2-Stage Pixal3D Pipeline with MoGe FOV estimation and mesh_scale optimization.

    Inherits all model components and inference methods from Pixal3DPipeline.
    Adds:
      - MoGe-based camera FOV estimation
      - Iterative mesh_scale optimization via dense inference loop
      - Optional separate dense_check model for mesh_scale optimization
        (uses a different dense checkpoint than the main dense model)
    """

    def __init__(
        self,
        *args,
        moge_model=None,
        dense_check_visual_condition=None,
        dense_check_denoiser_model=None,
        dense_check_scheduler=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.moge_model = moge_model
        # Optional separate dense model for mesh_scale optimization
        self.dense_check_visual_condition = dense_check_visual_condition
        self.dense_check_denoiser_model = dense_check_denoiser_model
        self.dense_check_scheduler = dense_check_scheduler

    @property
    def has_dense_check(self):
        """Whether a separate dense_check model is loaded."""
        return self.dense_check_denoiser_model is not None

    def _stage_module_names(self):
        stages = super()._stage_module_names()
        if self.moge_model is not None:
            stages["moge"] = ["moge_model"]
        if self.has_dense_check:
            stages["dense_check"] = [
                "dense_check_visual_condition",
                "dense_check_denoiser_model",
                "dense_vae",
            ]
        return stages

    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str = "./ckpt",
        repo_id: str = None,
        dense_dtype: torch.dtype = torch.float16,
        sparse_dtype: torch.dtype = torch.float16,
        cache_dir: str = None,
        use_moge: bool = True,
        moge_model_name: str = "Ruicheng/moge-vitl",
        use_dense_check: bool = True,
        load_device: str = "cpu",
        low_vram: bool = False,
    ):
        """
        Create Pixal3D 2-Stage Pipeline.

        Same as Pixal3DPipeline.from_pretrained but additionally loads:
          - MoGe model for FOV estimation
          - Optional dense_check model (separate dense dit for mesh_scale optimization,
            stored at dense/scale_init; scheduler & conditioner reuse dense/)

        Args:
            ckpt_dir: Local checkpoint directory for main models
            repo_id: HuggingFace repo ID for main models
            dense_dtype: Dense model dtype
            sparse_dtype: Sparse model dtype
            cache_dir: HF cache directory
            use_moge: Whether to load MoGe model for FOV estimation
            moge_model_name: MoGe model name on HuggingFace
            use_dense_check: Whether to load dense_check dit from dense/scale_init
            low_vram: Disable optional high-VRAM conditioner upsampling paths
        """
        import json
        import importlib
        from safetensors.torch import load_file

        # Use parent class to load all pipeline components
        base_pipeline = Pixal3DPipeline.from_pretrained(
            ckpt_dir=ckpt_dir,
            repo_id=repo_id,
            dense_dtype=dense_dtype,
            sparse_dtype=sparse_dtype,
            cache_dir=cache_dir,
            load_device=load_device,
            low_vram=low_vram,
        )

        # Load MoGe model
        moge_model = None
        if use_moge:
            moge_model = load_moge_model(device=load_device, model_name=moge_model_name)

        # Load dense_check dit (only the dit weights differ, scheduler & conditioner reuse dense/)
        dense_check_denoiser_model = None

        if use_dense_check:
            # Determine scale_init dit path
            scale_init_dit_loaded = False

            if repo_id is not None:
                from huggingface_hub import hf_hub_download
                try:
                    config_path = hf_hub_download(
                        repo_id=repo_id, subfolder="dense/scale_init",
                        filename="config.json", cache_dir=cache_dir, repo_type="model"
                    )
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    module_path, class_name = config["model_class"].rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    model_class = getattr(module, class_name)
                    exclude_keys = {"model_type", "model_class", "scheduler_class", "scheduler_config", "config"}
                    kwargs = {k: v for k, v in config.items() if k not in exclude_keys}
                    if hasattr(model_class, 'Config'):
                        dense_check_denoiser_model = model_class(cfg=kwargs)
                    else:
                        dense_check_denoiser_model = model_class(**kwargs)
                    safetensors_path = hf_hub_download(
                        repo_id=repo_id, subfolder="dense/scale_init",
                        filename="model.safetensors", cache_dir=cache_dir, repo_type="model"
                    )
                    state_dict = load_file(safetensors_path, device=load_device)
                    dense_check_denoiser_model.load_state_dict(state_dict, strict=True)
                    dense_check_denoiser_model.to(load_device).eval()
                    scale_init_dit_loaded = True
                    print("[2-Stage] dense_check dit loaded from HuggingFace (dense/scale_init)")
                except Exception as e:
                    print(f"[2-Stage] dense/scale_init not found on HF: {e}, trying local...")

            if not scale_init_dit_loaded:
                local_dit_dir = os.path.join(ckpt_dir, "dense", "scale_init")
                config_file = os.path.join(local_dit_dir, "config.json")
                safetensors_file = os.path.join(local_dit_dir, "model.safetensors")
                if os.path.exists(config_file) and os.path.exists(safetensors_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    module_path, class_name = config["model_class"].rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    model_class = getattr(module, class_name)
                    exclude_keys = {"model_type", "model_class", "scheduler_class", "scheduler_config", "config"}
                    kwargs = {k: v for k, v in config.items() if k not in exclude_keys}
                    if hasattr(model_class, 'Config'):
                        dense_check_denoiser_model = model_class(cfg=kwargs)
                    else:
                        dense_check_denoiser_model = model_class(**kwargs)
                    state_dict = load_file(safetensors_file, device=load_device)
                    dense_check_denoiser_model.load_state_dict(state_dict, strict=True)
                    dense_check_denoiser_model.to(load_device).eval()
                    print(f"[2-Stage] dense_check dit loaded from local: {local_dit_dir}")
                else:
                    print(f"[2-Stage] dense/scale_init not found locally, dense_check disabled")

        # Create 2-stage pipeline with all components from base
        # dense_check reuses base dense scheduler & conditioner
        pipeline = cls(
            dense_visual_condition=base_pipeline.dense_visual_condition,
            dense_denoiser_model=base_pipeline.dense_denoiser_model,
            dense_scheduler=base_pipeline.dense_scheduler,
            sparse_512_visual_condition=base_pipeline.sparse_512_visual_condition,
            sparse_512_denoiser_model=base_pipeline.sparse_512_denoiser_model,
            sparse_512_scheduler=base_pipeline.sparse_512_scheduler,
            sparse_1024_visual_condition=base_pipeline.sparse_1024_visual_condition,
            sparse_1024_denoiser_model=base_pipeline.sparse_1024_denoiser_model,
            sparse_1024_scheduler=base_pipeline.sparse_1024_scheduler,
            dense_vae=base_pipeline.dense_vae,
            sparse_vae_512=base_pipeline.sparse_vae_512,
            sparse_vae_1024=base_pipeline.sparse_vae_1024,
            dense_dtype=dense_dtype,
            sparse_dtype=sparse_dtype,
            moge_model=moge_model,
            dense_check_denoiser_model=dense_check_denoiser_model,
            # scheduler & conditioner reuse main dense
            dense_check_visual_condition=base_pipeline.dense_visual_condition if dense_check_denoiser_model else None,
            dense_check_scheduler=base_pipeline.dense_scheduler if dense_check_denoiser_model else None,
        )

        return pipeline

    def estimate_fov(self, image_path: str) -> float:
        """
        Estimate camera FOV from image using MoGe model.

        Args:
            image_path: Path to the preprocessed square image

        Returns:
            camera_angle_x in radians
        """
        if self.moge_model is None:
            raise ValueError("MoGe model not loaded. Set use_moge=True in from_pretrained().")
        self._ensure_stage("moge")
        try:
            return get_camera_angle_x_from_moge(image_path, self.moge_model, device=self.device)
        finally:
            self._offload_stage("moge")

    def _infer_dense_check(self, image, camera_angle_x, distance, mesh_scale, num_steps, guidance_scale, seed):
        """
        Run dense inference using the dense_check model (for mesh_scale optimization).
        Falls back to the main dense model if no dense_check model is loaded.
        """
        if not self.has_dense_check:
            return self.infer_dense(image, camera_angle_x, distance, mesh_scale, num_steps, guidance_scale, seed)

        self._ensure_stage("dense_check")

        from tqdm import tqdm

        batch_size = image.shape[0]
        do_cfg = guidance_scale > 0
        image = image.to(torch.float16)

        # Offload DiT and VAE to save VRAM for the visual condition / NAF upsampler!
        if self.dense_check_denoiser_model is not None:
            self.dense_check_denoiser_model.to("cpu")
        if getattr(self, "dense_vae", None) is not None:
            self.dense_vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Encode conditions using dense_check visual condition
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dense_dtype):
                cond_global, cond_proj = self.dense_check_visual_condition(
                    image[:, :3],
                    camera_angle_x=camera_angle_x,
                    distance=distance,
                    mesh_scale=mesh_scale,
                )
        uncond_global = torch.zeros_like(cond_global)
        uncond_proj = torch.zeros_like(cond_proj)
        cond = (cond_global, cond_proj)
        uncond = (uncond_global, uncond_proj)

        # Bring DiT and VAE back to GPU
        if self.dense_check_denoiser_model is not None:
            self.dense_check_denoiser_model.to(self.device)
        if getattr(self, "dense_vae", None) is not None:
            self.dense_vae.to(self.device)

        # Initialize latents
        latent_shape = (batch_size, *self.dense_check_denoiser_model.dit_model.latent_shape)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        latents = torch.randn(latent_shape, device=self.device, dtype=cond[0].dtype, generator=generator)

        # Setup scheduler
        self.dense_check_scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.dense_check_scheduler.timesteps
        extra_step_kwargs = {'generator': generator} if generator is not None else {}

        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Dense Check Sampling")):
            timestep_tensor = torch.tensor([t], dtype=latents.dtype, device=self.device)
            diffusion_inputs = {"x": latents, "t": timestep_tensor, "cond": cond}

            with torch.cuda.amp.autocast(dtype=self.dense_dtype):
                noise_pred_cond = self.dense_check_denoiser_model(**diffusion_inputs).sample
                if do_cfg:
                    diffusion_inputs["cond"] = uncond
                    noise_pred_uncond = self.dense_check_denoiser_model(**diffusion_inputs).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

            latents = self.dense_check_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        return latents

    def _optimize_mesh_scale(
        self,
        image_tensor: torch.Tensor,
        camera_angle_x_tensor: torch.Tensor,
        distance_tensor: torch.Tensor,
        initial_mesh_scale: float,
        dense_steps: int,
        dense_guidance_scale: float,
        dense_seed: int,
        dense_threshold: float,
        target_padding: int = 3,
        padding_tolerance_min: int = 2,
        padding_tolerance_max: int = 4,
        max_iterations: int = 2,
    ) -> tuple:
        """
        Iteratively optimize mesh_scale so decoded dense indices stay within grid boundaries.

        Uses the dense_check model (if available) for the optimization loop,
        then the main dense model is used for final inference in infer().

        Returns:
            optimized_mesh_scale (float)
        """
        current_mesh_scale = initial_mesh_scale
        best_mesh_scale = initial_mesh_scale

        use_check = self.has_dense_check
        check_label = "dense_check" if use_check else "dense"
        print(f"[mesh_scale optim] Using {check_label} model for optimization")

        # Initial dense inference with check model
        mesh_scale_tensor = torch.tensor([current_mesh_scale], device=self.device, dtype=torch.float32)
        dense_latents = self._infer_dense_check(
            image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor,
            dense_steps, dense_guidance_scale, dense_seed
        )
        with torch.autocast("cuda", dtype=torch.float16):
            decoded_index = self.dense_vae.decode_mesh(
                dense_latents, mc_threshold=dense_threshold, return_index=True
            )[0]
        decoded_index = sort_block(decoded_index, 8)

        for iteration in range(max_iterations):
            print(f"[mesh_scale optim] Iteration {iteration + 1}/{max_iterations}")

            optimal_mesh_scale = compute_optimal_mesh_scale(
                decoded_index=decoded_index,
                original_mesh_scale=current_mesh_scale,
                grid_resolution=64,
                padding=target_padding,
            )

            # Re-run dense inference with optimized mesh_scale (using check model)
            mesh_scale_tensor = torch.tensor([optimal_mesh_scale], device=self.device, dtype=torch.float32)
            dense_latents = self._infer_dense_check(
                image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor,
                dense_steps, dense_guidance_scale, dense_seed
            )
            with torch.autocast("cuda", dtype=torch.float16):
                opt_decoded_index = self.dense_vae.decode_mesh(
                    dense_latents, mc_threshold=dense_threshold, return_index=True
                )[0]
            opt_decoded_index = sort_block(opt_decoded_index, 8)

            # Check boundary
            xyz_index = opt_decoded_index[:, 1:4]
            min_padding = xyz_index.min(dim=0).values.min().item()
            max_padding = 63 - xyz_index.max(dim=0).values.max().item()
            actual_padding = min(min_padding, max_padding)

            print(f"[mesh_scale optim] mesh_scale={optimal_mesh_scale:.6f}, actual_padding={actual_padding}")

            if padding_tolerance_min <= actual_padding <= padding_tolerance_max:
                print(f"[mesh_scale optim] Padding {actual_padding} within [{padding_tolerance_min}, {padding_tolerance_max}], done!")
                best_mesh_scale = optimal_mesh_scale
                break
            elif actual_padding < padding_tolerance_min:
                print(f"[mesh_scale optim] Padding {actual_padding} < {padding_tolerance_min}, object too large, reverting")
                break
            else:
                print(f"[mesh_scale optim] Padding {actual_padding} > {padding_tolerance_max}, continuing...")
                best_mesh_scale = optimal_mesh_scale
                current_mesh_scale = optimal_mesh_scale
                decoded_index = opt_decoded_index
        else:
            print(f"[mesh_scale optim] Reached max iterations {max_iterations}, using best result")

        return best_mesh_scale

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
        mesh_scale: float = 0.5,
        optimize_mesh_scale: bool = True,
        target_padding: int = 3,
        max_optim_iterations: int = 2,
        mode_1024: str = "refine",
        target_face_count: int = 200000,
        remove_floaters: bool = True,
        remove_interior: bool = True,
    ):
        # Image preprocessing
        image_tensor = preprocess_image(image, 518, padding=20).unsqueeze(0).to(self.device)

        # Save preprocessed image for MoGe (MoGe needs the cropped/padded image, not the original)
        import tempfile
        img_np = (image_tensor[0, :3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(img_np).save(tmp_img.name)
        moge_image_path = tmp_img.name

        # Estimate camera FOV via MoGe
        camera_angle_x = self.estimate_fov(moge_image_path)

        # Clean up temporary file
        os.unlink(tmp_img.name)

        # Compute camera distance
        image_resolution = 518
        grid_points = torch.tensor([-1.0, 0, -1.0])
        grid_points = grid_points / mesh_scale / 2
        distance = distance_from_fov(
            camera_angle_x, grid_points,
            torch.tensor([0 - extend_pixel, image_resolution + extend_pixel]),
            mesh_scale, image_resolution
        )["distance_from_x"]

        print(f"[Pixal3D-2Stage] camera_angle_x: {camera_angle_x:.6f}, distance: {distance:.4f}, mesh_scale: {mesh_scale}")

        camera_angle_x_tensor = torch.tensor([camera_angle_x], device=self.device, dtype=torch.float32)
        distance_tensor = torch.tensor([distance], device=self.device, dtype=torch.float32)

        # ============ Step 1: Dense Inference + mesh_scale Optimization ============
        if optimize_mesh_scale:
            print(f"[Pixal3D-2Stage] Step 1: mesh_scale optimization (using {'dense_check' if self.has_dense_check else 'dense'} model)...")
            mesh_scale = self._optimize_mesh_scale(
                image_tensor, camera_angle_x_tensor, distance_tensor,
                initial_mesh_scale=mesh_scale,
                dense_steps=dense_steps,
                dense_guidance_scale=dense_guidance_scale,
                dense_seed=dense_seed,
                dense_threshold=dense_threshold,
                target_padding=target_padding,
                max_iterations=max_optim_iterations,
            )
            print(f"[Pixal3D-2Stage] Optimized mesh_scale: {mesh_scale:.6f}")

        # Always run final dense inference with the MAIN dense model
        print(f"[Pixal3D-2Stage] Step 1b: Final Dense Inference with main dense model...")
        mesh_scale_tensor = torch.tensor([mesh_scale], device=self.device, dtype=torch.float32)
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

        print(f"[Pixal3D-2Stage] decoded_index shape: {decoded_index.shape}, "
              f"max: {decoded_index.max(0).values.tolist()}, min: {decoded_index.min(0).values.tolist()}")

        print(f"[Pixal3D-2Stage] Step 2: Sparse 512 Inference...")
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
        print(f"[VRAM] after 512 cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB alloc, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

        rotation_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        if mode_1024 == "skip":
            print(f"[Pixal3D-2Stage] mode_1024=skip, returning 512 mesh directly.")
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
            return _meshlib_postprocess(result, target_face_count, remove_floaters, remove_interior)

        print(f"[Pixal3D-2Stage] Step 3: Prepare 1024 latent index...")
        latent_index_1024 = mesh2index(mesh_512, size=1024, factor=8)
        del mesh_512
        gc.collect()
        torch.cuda.empty_cache()
        block_size_1024 = getattr(self.sparse_1024_denoiser_model.dit_model, 'selection_block_size', 8)
        latent_index_1024 = sort_block(latent_index_1024, block_size_1024)
        print(f"[Pixal3D-2Stage] 1024 latent tokens: {len(latent_index_1024)}")
        print(f"[VRAM] before 1024 inference: {torch.cuda.memory_allocated()/1024**3:.2f} GB alloc, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

        print(f"[Pixal3D-2Stage] Step 4: Sparse 1024 Inference (mode={mode_1024})...")
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

        return _meshlib_postprocess(mesh_1024, target_face_count, remove_floaters, remove_interior)

    def infer_from_image(self, image_path: str, **kwargs):
        return self.infer(image=image_path, **kwargs)
