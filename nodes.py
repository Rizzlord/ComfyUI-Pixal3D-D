import os
import torch
import numpy as np
from PIL import Image
import trimesh
import folder_paths
import comfy.model_management as mm
import gc

# Import the pipeline and utilities
from .pixal3dpipeline2stage import Pixal3DPipeline2Stage
from .pixal3dpipeline import _meshlib_postprocess, preprocess_image
from pixal3d.utils import sort_block, mesh2index, normalize_mesh


def _mesh_counts(mesh):
    return len(getattr(mesh, "vertices", [])), len(getattr(mesh, "faces", []))


def _is_valid_mesh(mesh):
    verts, faces = _mesh_counts(mesh)
    return verts > 0 and faces > 0


def _scale_and_center_mesh(mesh, mesh_scale, label):
    verts = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    if verts.size == 0 or not np.isfinite(verts).all():
        print(f"{label}: mesh has non-finite vertices, skipping scale/center")
        return mesh
    mesh.apply_scale(0.5 / mesh_scale)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape == (2, 3) and np.isfinite(bounds).all():
        mesh.vertices -= bounds.mean(axis=0)
    else:
        print(f"{label}: skipping centering due to non-finite bounds {bounds}")
    return mesh


class Pixal3DLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": "If False, the model will be moved to CPU and VRAM cleared after generation"}),
                "low_vram": ("BOOLEAN", {"default": False, "tooltip": "Disable the optional NAF upsampler in visual conditioning to reduce VRAM usage on 16GB GPUs"}),
                "upsample_res": (["518", "256", "128"], {"default": "518", "tooltip": "Resolution for NAF upsampler. Lower values save significant VRAM."}),
                "chunk_encoding": ("BOOLEAN", {"default": False, "tooltip": "Process views one by one during encoding to save VRAM."}),
            },
        }

    RETURN_TYPES = ("PIXAL3D_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "Pixal3D-D"

    def load_model(self, keep_model_loaded, low_vram, upsample_res="518", chunk_encoding=False):
        ckpt_path = os.path.join(folder_paths.models_dir, "pixal3d-d")
        
        if not os.path.exists(os.path.join(ckpt_path, "dense")):
            raise FileNotFoundError(f"Pixal3D-D models not found in {ckpt_path}. Please ensure 'dense', 'sparse512', and 'sparse1024' folders are present.")
            
        device = mm.get_torch_device()
        pipeline = Pixal3DPipeline2Stage.from_pretrained(
            ckpt_dir=ckpt_path,
            dense_dtype=torch.float16,
            sparse_dtype=torch.float16,
            low_vram=low_vram,
        )
        pipeline.keep_model_loaded = keep_model_loaded
        pipeline.set_upsample_res(int(upsample_res))
        pipeline.set_chunk_encoding(chunk_encoding)
        pipeline.enable_model_cpu_offload(device=device)
        return (pipeline,)

class Pixal3DGenerateDense:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIXAL3D_PIPELINE",),
                "image": ("IMAGE", {"tooltip": "Reference image for 3D generation"}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "tooltip": "Number of denoising steps for the initial dense stage"}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "tooltip": "Classifier-free guidance scale"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "tooltip": "Random seed for generation"}),
                "mesh_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "tooltip": "Initial mesh scale factor"}),
                "optimize_mesh_scale": ("BOOLEAN", {"default": True, "tooltip": "Automatically adjust mesh scale to fit the grid perfectly"}),
                "dense_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Marching cubes threshold for the dense stage"}),
            },
        }

    RETURN_TYPES = ("PIXAL3D_LATENT_INDEX", "FLOAT")
    RETURN_NAMES = ("latent_index", "mesh_scale")
    FUNCTION = "generate"
    CATEGORY = "Pixal3D-D"

    def generate(self, pipeline, image, steps, guidance_scale, seed, mesh_scale, optimize_mesh_scale, dense_threshold):
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        image_tensor = preprocess_image(img, 518, padding=20).unsqueeze(0).to(pipeline.device)
        
        import tempfile
        img_np = (image_tensor[0, :3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            Image.fromarray(img_np).save(tmp_img.name)
            camera_angle_x = pipeline.estimate_fov(tmp_img.name)
        os.unlink(tmp_img.name)

        from .pixal3dpipeline import distance_from_fov
        grid_points = torch.tensor([-1.0, 0, -1.0]) / mesh_scale / 2
        distance = distance_from_fov(camera_angle_x, grid_points, torch.tensor([0 - 20, 518 + 20]), mesh_scale, 518)["distance_from_x"]

        camera_angle_x_tensor = torch.tensor([camera_angle_x], device=pipeline.device, dtype=torch.float32)
        distance_tensor = torch.tensor([distance], device=pipeline.device, dtype=torch.float32)

        if optimize_mesh_scale:
            mesh_scale = pipeline._optimize_mesh_scale(
                image_tensor, camera_angle_x_tensor, distance_tensor,
                initial_mesh_scale=mesh_scale,
                dense_steps=steps, dense_guidance_scale=guidance_scale,
                dense_seed=seed, dense_threshold=dense_threshold
            )
        
        mesh_scale_tensor = torch.tensor([mesh_scale], device=pipeline.device, dtype=torch.float32)
        dense_latents = pipeline.infer_dense(
            image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor,
            steps, guidance_scale, seed
        )
        with torch.autocast("cuda", dtype=torch.float16):
            decoded_index = pipeline.dense_vae.decode_mesh(dense_latents, mc_threshold=dense_threshold, return_index=True)[0]
        
        del dense_latents
        decoded_index = sort_block(decoded_index, 8)
        
        ctx = {
            "index": decoded_index,
            "image_tensor": image_tensor.cpu(),
            "camera_angle_x": camera_angle_x,
            "distance": distance,
            "mesh_scale": mesh_scale
        }
        
        pipeline._offload_stage("dense")
        gc.collect()
        torch.cuda.empty_cache()
        
        return (ctx, mesh_scale)

class Pixal3DRefineSparse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIXAL3D_PIPELINE",),
                "latent_index": ("PIXAL3D_LATENT_INDEX", {"tooltip": "Output from the Generate Dense Index node"}),
                "mode_1024": (["full", "refine", "skip"], {"default": "skip", "tooltip": "1024 refinement mode. 'full' generates native 1024 sparse conditioning tokens"}),
                "sparse_512_steps": ("INT", {"default": 30, "min": 1, "max": 200, "tooltip": "Steps for the 512 sparse refinement stage"}),
                "sparse_1024_steps": ("INT", {"default": 15, "min": 1, "max": 200, "tooltip": "Steps for the 1024 sparse refinement stage"}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "tooltip": "Classifier-free guidance scale"}),
                "mc_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Marching cubes threshold for final mesh extraction"}),
                "target_face_count": ("INT", {"default": 5000000, "min": 0, "max": 5000000, "tooltip": "Target number of faces for decimation (0 to skip)"}),
                "remove_floaters": ("BOOLEAN", {"default": True, "tooltip": "Remove disconnected small components (floaters)"}),
                "remove_interior": ("BOOLEAN", {"default": True, "tooltip": "Remove internal shells and fully enclosed geometry"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "generate"
    CATEGORY = "Pixal3D-D"

    def generate(self, pipeline, latent_index, mode_1024, sparse_512_steps, sparse_1024_steps, 
                 guidance_scale, mc_threshold, target_face_count, remove_floaters, remove_interior, seed):
        
        ctx = latent_index
        pipeline._offload_stage("dense") # Ensure dense is gone
        index = ctx["index"].to(pipeline.device)
        image_tensor = ctx["image_tensor"].to(pipeline.device)
        camera_angle_x = ctx["camera_angle_x"]
        distance = ctx["distance"]
        mesh_scale = ctx["mesh_scale"]
        
        camera_angle_x_tensor = torch.tensor([camera_angle_x], device=pipeline.device, dtype=torch.float32)
        distance_tensor = torch.tensor([distance], device=pipeline.device, dtype=torch.float32)
        mesh_scale_tensor = torch.tensor([mesh_scale], device=pipeline.device, dtype=torch.float32)

        sparse_latents_512 = pipeline.infer_sparse(
            image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor, index,
            sparse_512_steps, guidance_scale, seed,
            pipeline.sparse_512_visual_condition, pipeline.sparse_512_denoiser_model, pipeline.sparse_512_scheduler
        )

        pipeline._offload_stage("sparse512_dit")
        pipeline._ensure_stage("sparse512_vae")
        with torch.autocast("cuda", dtype=torch.float16):
            mesh_512 = pipeline.sparse_vae_512.decode_mesh(sparse_latents_512, voxel_resolution=512)[0]
        
        del index, sparse_latents_512
        pipeline._offload_stage("sparse512_vae")
        gc.collect()
        torch.cuda.empty_cache()

        if mode_1024 == "skip":
            from pixal3d.utils import postprocess_mesh
            mesh_v, mesh_f = postprocess_mesh(mesh_512.vertices, mesh_512.faces, simplify=False)
            res = trimesh.Trimesh(mesh_v, mesh_f, process=False)
            res = _scale_and_center_mesh(res, mesh_scale, "[Pixal3DRefineSparse] 512 mesh")
            if not pipeline.keep_model_loaded:
                pipeline.unload()
                mm.soft_empty_cache()
            else:
                pipeline.offload_all_models()
            return (_meshlib_postprocess(res, target_face_count, remove_floaters, remove_interior),)

        latent_index_1024 = mesh2index(mesh_512, size=1024, factor=8).to(pipeline.device)
        latent_index_1024 = sort_block(latent_index_1024, 8)

        cross_res = None
        if mode_1024 == "refine":
            cross_res = (pipeline.sparse_512_visual_condition or pipeline.dense_visual_condition, "sparse512_cond", 128)

        vis_1024 = pipeline.sparse_1024_visual_condition or pipeline.sparse_512_visual_condition or pipeline.dense_visual_condition

        sparse_latents_1024 = pipeline.infer_sparse(
            image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor, latent_index_1024,
            sparse_1024_steps, guidance_scale, seed,
            vis_1024, pipeline.sparse_1024_denoiser_model, pipeline.sparse_1024_scheduler,
            cross_res_cond=cross_res
        )

        pipeline._offload_stage("sparse1024_dit")
        pipeline._ensure_stage("sparse1024_vae")
        with torch.autocast("cuda", dtype=torch.float16):
            decoded_mesh_1024 = pipeline.sparse_vae_1024.decode_mesh(sparse_latents_1024, voxel_resolution=1024, mc_threshold=mc_threshold)[0]

        if mode_1024 == "refine" and not _is_valid_mesh(decoded_mesh_1024):
            print("[Pixal3DRefineSparse] refine mode produced an empty 1024 mesh, retrying with native 1024 conditioning.")
            del decoded_mesh_1024, sparse_latents_1024
            pipeline._offload_stage("sparse1024_vae")
            gc.collect()
            torch.cuda.empty_cache()
            fb_cross_res = cross_res if pipeline.sparse_1024_visual_condition is None else None
            sparse_latents_1024 = pipeline.infer_sparse(
                image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor, latent_index_1024,
                sparse_1024_steps, guidance_scale, seed,
                vis_1024, pipeline.sparse_1024_denoiser_model, pipeline.sparse_1024_scheduler,
                cross_res_cond=fb_cross_res
            )
            pipeline._offload_stage("sparse1024_dit")
            pipeline._ensure_stage("sparse1024_vae")
            with torch.autocast("cuda", dtype=torch.float16):
                decoded_mesh_1024 = pipeline.sparse_vae_1024.decode_mesh(
                    sparse_latents_1024, voxel_resolution=1024, mc_threshold=mc_threshold
                )[0]

        if _is_valid_mesh(decoded_mesh_1024):
            from pixal3d.utils import postprocess_mesh
            mesh_v, mesh_f = postprocess_mesh(decoded_mesh_1024.vertices, decoded_mesh_1024.faces, simplify=False)
            res = trimesh.Trimesh(mesh_v, mesh_f, process=False)
            res = _scale_and_center_mesh(res, mesh_scale, "[Pixal3DRefineSparse] 1024 mesh")
        else:
            print("[Pixal3DRefineSparse] 1024 decode still empty after fallback, returning the 512 mesh instead.")
            from pixal3d.utils import postprocess_mesh
            mesh_v, mesh_f = postprocess_mesh(mesh_512.vertices, mesh_512.faces, simplify=False)
            res = trimesh.Trimesh(mesh_v, mesh_f, process=False)
            res = _scale_and_center_mesh(res, mesh_scale, "[Pixal3DRefineSparse] 512 fallback mesh")
        
        if not pipeline.keep_model_loaded:
            pipeline.unload()
            mm.soft_empty_cache()
        else:
            pipeline.offload_all_models()
        return (_meshlib_postprocess(res, target_face_count, remove_floaters, remove_interior),)

class Pixal3DRefineMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIXAL3D_PIPELINE",),
                "image": ("IMAGE", {"tooltip": "Reference image for refinement context"}),
                "mesh": ("TRIMESH", {"tooltip": "Existing mesh (e.g. from GLB) to enhance"}),
                "mode_1024": (["full", "refine"], {"default": "full", "tooltip": "'full' generates native 1024 sparse conditioning tokens"}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 200, "tooltip": "Refinement steps"}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0}),
                "mc_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_face_count": ("INT", {"default": 5000000, "min": 0, "max": 5000000}),
                "remove_floaters": ("BOOLEAN", {"default": True, "tooltip": "Remove disconnected small components (floaters)"}),
                "remove_interior": ("BOOLEAN", {"default": True, "tooltip": "Remove internal geometry using meshlib"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "generate"
    CATEGORY = "Pixal3D-D"

    def generate(self, pipeline, image, mesh, mode_1024, steps, guidance_scale, 
                 mc_threshold, target_face_count, remove_floaters, remove_interior, seed):
        
        pipeline._offload_stage("dense")
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        image_tensor = preprocess_image(img, 518, padding=20).unsqueeze(0).to(pipeline.device)
        
        import tempfile
        img_np = (image_tensor[0, :3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            Image.fromarray(img_np).save(tmp_img.name)
            camera_angle_x = pipeline.estimate_fov(tmp_img.name)
        os.unlink(tmp_img.name)

        mesh_to_refine = mesh.copy()
        mesh_to_refine = normalize_mesh(mesh_to_refine, scale=0.95)
        latent_index_1024 = mesh2index(mesh_to_refine, size=1024, factor=8).to(pipeline.device)
        latent_index_1024 = sort_block(latent_index_1024, 8)

        original_mesh = mesh.copy()
        from .pixal3dpipeline import distance_from_fov
        mesh_scale = 0.95
        grid_points = torch.tensor([-1.0, 0, -1.0]) / mesh_scale / 2
        distance = distance_from_fov(camera_angle_x, grid_points, torch.tensor([0 - 20, 518 + 20]), mesh_scale, 518)["distance_from_x"]

        camera_angle_x_tensor = torch.tensor([camera_angle_x], device=pipeline.device, dtype=torch.float32)
        distance_tensor = torch.tensor([distance], device=pipeline.device, dtype=torch.float32)
        mesh_scale_tensor = torch.tensor([mesh_scale], device=pipeline.device, dtype=torch.float32)

        cross_res = None
        if mode_1024 == "refine":
            cross_res = (pipeline.sparse_512_visual_condition or pipeline.dense_visual_condition, "sparse512_cond", 128)

        vis_1024 = pipeline.sparse_1024_visual_condition or pipeline.sparse_512_visual_condition or pipeline.dense_visual_condition

        sparse_latents_1024 = pipeline.infer_sparse(
            image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor, latent_index_1024,
            steps, guidance_scale, seed,
            vis_1024, pipeline.sparse_1024_denoiser_model, pipeline.sparse_1024_scheduler,
            cross_res_cond=cross_res
        )

        pipeline._offload_stage("sparse1024_dit")
        pipeline._ensure_stage("sparse1024_vae")
        with torch.autocast("cuda", dtype=torch.float16):
            decoded_mesh_1024 = pipeline.sparse_vae_1024.decode_mesh(sparse_latents_1024, voxel_resolution=1024, mc_threshold=mc_threshold)[0]

        if mode_1024 == "refine" and not _is_valid_mesh(decoded_mesh_1024):
            print("[Pixal3DRefineMesh] refine mode produced an empty 1024 mesh, retrying with native 1024 conditioning.")
            del decoded_mesh_1024, sparse_latents_1024
            pipeline._offload_stage("sparse1024_vae")
            gc.collect()
            torch.cuda.empty_cache()
            fb_cross_res = cross_res if pipeline.sparse_1024_visual_condition is None else None
            sparse_latents_1024 = pipeline.infer_sparse(
                image_tensor, camera_angle_x_tensor, distance_tensor, mesh_scale_tensor, latent_index_1024,
                steps, guidance_scale, seed,
                vis_1024, pipeline.sparse_1024_denoiser_model, pipeline.sparse_1024_scheduler,
                cross_res_cond=fb_cross_res
            )
            pipeline._offload_stage("sparse1024_dit")
            pipeline._ensure_stage("sparse1024_vae")
            with torch.autocast("cuda", dtype=torch.float16):
                decoded_mesh_1024 = pipeline.sparse_vae_1024.decode_mesh(
                    sparse_latents_1024, voxel_resolution=1024, mc_threshold=mc_threshold
                )[0]

        if _is_valid_mesh(decoded_mesh_1024):
            from pixal3d.utils import postprocess_mesh
            mesh_v, mesh_f = postprocess_mesh(decoded_mesh_1024.vertices, decoded_mesh_1024.faces, simplify=False)
            res = trimesh.Trimesh(mesh_v, mesh_f, process=False)
            res = _scale_and_center_mesh(res, mesh_scale, "[Pixal3DRefineMesh] 1024 mesh")
        else:
            print("[Pixal3DRefineMesh] 1024 decode still empty after fallback, returning the input mesh instead.")
            res = original_mesh
        
        if not pipeline.keep_model_loaded:
            pipeline.unload()
            mm.soft_empty_cache()
        else:
            pipeline.offload_all_models()
        return (_meshlib_postprocess(res, target_face_count, remove_floaters, remove_interior),)

NODE_CLASS_MAPPINGS = {
    "Pixal3DLoader": Pixal3DLoader,
    "Pixal3DGenerateDense": Pixal3DGenerateDense,
    "Pixal3DRefineSparse": Pixal3DRefineSparse,
    "Pixal3DRefineMesh": Pixal3DRefineMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DLoader": "Pixal3D-D Model Loader",
    "Pixal3DGenerateDense": "Pixal3D-D Generate Dense Index",
    "Pixal3DRefineSparse": "Pixal3D-D Refine Sparse Latents",
    "Pixal3DRefineMesh": "Pixal3D-D Refine Existing Mesh",
}
