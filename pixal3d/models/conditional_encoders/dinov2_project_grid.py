"""
DINOv2 Project Grid Encoders
Includes single-view and multi-view DINOv2 encoders with 3D grid projection support
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from diffusers.models.modeling_utils import ModelMixin

import pixal3d
from pixal3d.utils.base import BaseModule

# Set linear algebra backend to avoid cusolver errors
try:
    torch.backends.cuda.preferred_linalg_library("cusolver")
except Exception:
    pass


# =============================================================================
# Base DINOv2 Encoder
# =============================================================================

@pixal3d.register("dinov2-encoder")
class DinoEncoder(BaseModule, ModelMixin):
    """Base DINOv2 Encoder"""

    @dataclass
    class Config(BaseModule.Config):
        model: str = "facebookresearch/dinov2"
        version: str = "dinov2_vitl14_reg"
        size: int = 518
        empty_embeds_ratio: float = 0.1

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.empty_embeds_ratio = self.cfg.empty_embeds_ratio

        # Load DINOv2 model
        dino_model = torch.hub.load(
            self.cfg.model, self.cfg.version, pretrained=True
        )
        self.encoder = dino_model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(
                self.cfg.size,
                transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
            transforms.CenterCrop(self.cfg.size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

      


    def forward(self, image, image_mask=None, is_training=False):
        z = self.encoder(self.transform(image), is_training=True)['x_prenorm']
        z = F.layer_norm(z, z.shape[-1:])

        if is_training and random.random() < self.empty_embeds_ratio:
            # zero out embeddings
            z = z * 0

        if image_mask is not None:
            image_mask_patch = F.max_pool2d(
                image_mask, kernel_size=14, stride=14
            ).squeeze(1) > 0
            return z, image_mask_patch

        return z


# =============================================================================
# 3D Projection Utility Functions
# =============================================================================

def project_points_to_image_batch(
    points_3d: torch.Tensor,
    transform_matrix: torch.Tensor,
    camera_angle_x: torch.Tensor,
    resolution: int = 518
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D image coordinates with batch support

    Args:
        points_3d: [N, 3] or [B, N, 3], 3D point coordinates (in [-1, 1] range)
        transform_matrix: [B, 4, 4], batch of camera transformation matrices
        camera_angle_x: [B], batch of camera horizontal FOV angles (radians)
        resolution: Rendering image resolution

    Returns:
        points_2d: [B, N, 2], image coordinates [x, y]
        depth: [B, N], depth values
        valid_mask: [B, N], mask indicating if points are within view
    """
    device = points_3d.device
    B = transform_matrix.shape[0]

    # Ensure inputs are torch.Tensor
    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(
            transform_matrix, dtype=torch.float32, device=device
        )
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(
            points_3d, dtype=torch.float32, device=device
        )
    if not isinstance(camera_angle_x, torch.Tensor):
        camera_angle_x = torch.tensor(
            camera_angle_x, dtype=torch.float32, device=device
        )

    # Expand points_3d to batch dimension
    if points_3d.dim() == 2:
        points_3d_batch = points_3d.unsqueeze(0).expand(B, -1, -1)
    else:
        points_3d_batch = points_3d

    N = points_3d_batch.shape[1]

    # Add homogeneous coordinates
    ones = torch.ones(B, N, 1, device=device)
    points_homogeneous = torch.cat([points_3d_batch, ones], dim=-1)

    # World to camera transformation
    world_to_camera = torch.linalg.inv(transform_matrix)
    points_camera = torch.bmm(
        points_homogeneous,
        world_to_camera.transpose(-2, -1)
    )[..., :3]

    # Extract camera coordinates
    x_cam = points_camera[..., 0]
    y_cam = points_camera[..., 1]
    z_cam = points_camera[..., 2]

    # Depth values
    depth = -z_cam

    # Compute camera intrinsics
    sensor_width = 32.0
    focal_length = 16.0 / torch.tan(camera_angle_x / 2.0)
    focal_length_pixels = focal_length * resolution / sensor_width
    focal_length_pixels = focal_length_pixels.unsqueeze(1)

    # Perspective projection
    x_ndc = focal_length_pixels * x_cam / (-z_cam)
    y_ndc = focal_length_pixels * y_cam / (-z_cam)

    # Convert to image coordinates
    x_pixel = x_ndc + resolution / 2.0
    y_pixel = -y_ndc + resolution / 2.0

    # Validity mask
    valid_mask = (
        (x_pixel >= 0) & (x_pixel < resolution) &
        (y_pixel >= 0) & (y_pixel < resolution) &
        (depth > 0)
    )

    points_2d = torch.stack([x_pixel, y_pixel], dim=-1)
    return points_2d, depth, valid_mask


def project_points_to_image(
    points_3d: torch.Tensor,
    transform_matrix: torch.Tensor,
    camera_angle_x: float,
    resolution: int = 512
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D image coordinates (single-view version)

    Args:
        points_3d: [N, 3], 3D point coordinates
        transform_matrix: [4, 4], camera transformation matrix
        camera_angle_x: Camera horizontal FOV angle (radians)
        resolution: Rendering image resolution

    Returns:
        points_2d: [N, 2], image coordinates [x, y]
        depth: [N], depth values
        valid_mask: [N], mask indicating if points are within view
    """
    device = points_3d.device

    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(
            transform_matrix, dtype=torch.float32, device=device
        )
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(
            points_3d, dtype=torch.float32, device=device
        )

    N = points_3d.shape[0]
    points_homogeneous = torch.cat([
        points_3d,
        torch.ones(N, 1, device=device)
    ], dim=1)

    # World to camera transformation
    camera_to_world = transform_matrix
    world_to_camera = torch.linalg.inv(camera_to_world)
    points_camera = torch.matmul(
        points_homogeneous,
        world_to_camera.T
    )[:, :3]

    x_cam = points_camera[:, 0]
    y_cam = points_camera[:, 1]
    z_cam = points_camera[:, 2]
    depth = -z_cam

    # Camera intrinsics
    sensor_width = 32.0
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    focal_length_pixels = focal_length * resolution / sensor_width

    # Perspective projection
    x_ndc = focal_length_pixels * x_cam / (-z_cam)
    y_ndc = focal_length_pixels * y_cam / (-z_cam)

    # Image coordinates
    x_pixel = x_ndc + resolution / 2.0
    y_pixel = -y_ndc + resolution / 2.0

    valid_mask = (
        (x_pixel >= 0) & (x_pixel < resolution) &
        (y_pixel >= 0) & (y_pixel < resolution) &
        (depth > 0)
    )

    points_2d = torch.stack([x_pixel, y_pixel], dim=1)
    return points_2d, depth, valid_mask


def sample_features(
    fmap: torch.Tensor,
    queries_ndc: torch.Tensor
) -> torch.Tensor:
    """
    Sample features using grid_sample

    Args:
        fmap: [B, C, H, W], feature map
        queries_ndc: [B, K, 2], NDC coordinates

    Returns:
        feat: [B, C, K], sampled features
    """
    B, C, H, W = fmap.shape
    Bq, K, _ = queries_ndc.shape
    assert Bq == B, "batch 不一致"

    grid = queries_ndc.view(B, K, 1, 2)
    feat = F.grid_sample(
        fmap, grid, mode='bilinear',
        align_corners=False, padding_mode='border'
    )
    return feat.squeeze(-1)


# =============================================================================
# Projection Grid Module
# =============================================================================

class ProjGrid(nn.Module):
    """3D Grid Projection Module"""

    def __init__(self, grid_resolution: int = 16):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.image_resolution = 518

        # Create 3D grid points
        one_dim = torch.linspace(-1, 1, grid_resolution)
        x, y, z = torch.meshgrid(one_dim, one_dim, one_dim, indexing='ij')
        grid_points = torch.stack((x, y, z), dim=-1)

        # Rotation matrix (align with Blender)
        rotation_matrix = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ])
        grid_points = torch.matmul(grid_points, rotation_matrix.T)
        grid_points = grid_points.reshape(-1, 3)
        self.register_buffer('grid_points', grid_points)

        # Front view transformation matrix
        front_view_transform_matrix = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.register_buffer(
            "front_view_transform_matrix",
            front_view_transform_matrix
        )

    def forward(
        self,
        features_map: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: torch.Tensor = None,
        BHWC: bool = True
    ) -> torch.Tensor:
        """
        Project feature map to 3D grid

        Args:
            features_map: [B, H, W, C] or [B, C, H, W]
            camera_angle_x: [B]
            distance: [B]
            mesh_scale: [B]
            transform_matrix: [B, 4, 4] or None
            BHWC: Whether input is in BHWC format

        Returns:
            x: [B, K, C], projected features
        """
        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape

        # Prepare grid points
        grid_points = self.grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2

        # Use default transformation matrix
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance

        # Project to image
        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )

        # Normalize to [-1, 1]
        
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1
   

        # Adjust dimensions and sample
        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)

        x = sample_features(features_map, image_points_norm)
        x = x.permute(0, 2, 1)

        return x





# =============================================================================
# DINOv2 Encoder with Projection
# =============================================================================

@pixal3d.register("dinov2-encoder-proj")
class DinoEncoderProj(BaseModule, ModelMixin):
    """DINOv2 Encoder with 3D Grid Projection"""

    @dataclass
    class Config(BaseModule.Config):
        model: str = "facebookresearch/dinov2"
        version: str = "dinov2_vitl14_reg"
        size: int = 518
        empty_embeds_ratio: float = 0.1
        grid_resolution: int = 16
        use_upsample: bool = False
        use_geo_feats: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.grid_resolution = self.cfg.grid_resolution
        self.empty_embeds_ratio = self.cfg.empty_embeds_ratio
        self.use_upsample = self.cfg.use_upsample

        # Load DINOv2
        dino_model = torch.hub.load(
            self.cfg.model, self.cfg.version, pretrained=True
        )
        self.encoder = dino_model.eval()

        # Optional: load upsampler
        if self.use_upsample:
            upsampler = torch.hub.load("valeoai/NAF", "naf", pretrained=True)
            self.upsampler = upsampler.eval()

        # Image preprocessing (normalization only)
        self.transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.patch_size = self.encoder.patch_size
        self.patch_number = self.cfg.size // self.patch_size
        self.proj_grid = ProjGrid(grid_resolution=self.cfg.grid_resolution)


     

        

    def forward(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor = None,
        camera_angle_x: torch.Tensor = None,
        distance: torch.Tensor = None,
        mesh_scale: torch.Tensor = None,
        transform_matrix: torch.Tensor = None,
        is_training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            image: [B, C, H, W]
            camera_angle_x: [B]
            distance: [B]
            mesh_scale: [B]
            is_training: Training mode flag

        Returns:
            z_global: [B, num_global, C]
            z: [B, grid_resolution^3, C]
        """
        image = self.transform(image)

        with torch.no_grad():
            z = self.encoder(image, is_training=True)['x_prenorm']
            z = F.layer_norm(z, z.shape[-1:])

            # Split tokens
            z_clstoken = z[:, 0:1]
            z_regtokens = z[:, 1:self.encoder.num_register_tokens + 1]
            z_patchtokens = z[:, 1 + self.encoder.num_register_tokens:]
            z_patchtokens = z_patchtokens.reshape(
                z_patchtokens.shape[0],
                self.patch_number,
                self.patch_number,
                -1
            )

            # Project to grid
            z = self.proj_grid(
                z_patchtokens, camera_angle_x, distance, mesh_scale
            )

            # Optional: upsample and fuse
            if self.use_upsample:
                z_patchtokens_permuted = z_patchtokens.permute(0, 3, 1, 2)
                z_upsampled = self.upsampler(
                    image, z_patchtokens_permuted, output_size=(518, 518)
                )
                z_upsampled = self.proj_grid(
                    z_upsampled, camera_angle_x, distance, mesh_scale, BHWC=False
                )
                z = z + z_upsampled

        # Global tokens
        z_global = torch.cat([z_clstoken, z_regtokens], dim=1)
        z_global = z_global.expand(z.shape[0], -1, -1)

        # Classifier-free guidance: random drop
        if is_training and random.random() < self.empty_embeds_ratio:
            z_global = z_global * 0
            z = z * 0

        return z_global, z


# =============================================================================
# Multi-View Projection Encoder Helper Functions
# =============================================================================

def compute_calc_mat(
    true_view_mat: torch.Tensor,
    ext_true_view_mat: torch.Tensor,
    fix_mat: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute calc_mat using matrix relative transformation

    Args:
        true_view_mat: [B, 1, 4, 4], ground truth camera matrix
        ext_true_view_mat: [B, N, 4, 4], extended ground truth camera matrices
        fix_mat: [B, 1, 4, 4], fixed matrix

    Returns:
        calc_mat: [B, N, 4, 4]
        relative_transform: [B, N, 4, 4]
    """
    B, N = ext_true_view_mat.shape[:2]

    # Expand to [B, N, 4, 4]
    true_view_mat_exp = true_view_mat.expand(B, N, 4, 4)
    fix_mat_exp = fix_mat.expand(B, N, 4, 4)

    # Flatten to [B*N, 4, 4]
    true_view_mat_flat = true_view_mat_exp.reshape(B * N, 4, 4)
    ext_true_view_mat_flat = ext_true_view_mat.reshape(B * N, 4, 4)
    fix_mat_flat = fix_mat_exp.reshape(B * N, 4, 4)

    # Compute relative transformation (disable autocast for fp32 precision)
    with torch.amp.autocast('cuda', enabled=False):
        true_view_mat_flat = true_view_mat_flat.float()
        ext_true_view_mat_flat = ext_true_view_mat_flat.float()
        fix_mat_flat = fix_mat_flat.float()

        relative_transform_flat = torch.bmm(
            torch.linalg.inv(true_view_mat_flat),
            ext_true_view_mat_flat
        )
        calc_mat_flat = torch.bmm(fix_mat_flat, relative_transform_flat)

    calc_mat = calc_mat_flat.view(B, N, 4, 4)
    relative_transform = relative_transform_flat.view(B, N, 4, 4)

    return calc_mat, relative_transform


# =============================================================================
# Multi-View DINOv2 Projection Encoder
# =============================================================================

@pixal3d.register("dinov2-encoder-proj-multi-view")
class DinoEncoderProjMultiView(BaseModule, ModelMixin):
    """Multi-View DINOv2 Projection Encoder"""

    @dataclass
    class Config(BaseModule.Config):
        model: str = "facebookresearch/dinov2"
        version: str = "dinov2_vitl14_reg"
        size: int = 518
        empty_embeds_ratio: float = 0.1
        grid_resolution: int = 16
        use_upsample: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.grid_resolution = self.cfg.grid_resolution
        self.empty_embeds_ratio = self.cfg.empty_embeds_ratio
        self.use_upsample = self.cfg.use_upsample

        # Load DINOv2
        dino_model = torch.hub.load(
            self.cfg.model, self.cfg.version, pretrained=True
        )
    
        self.encoder = dino_model.eval()

        # Optional: upsampler
        if self.use_upsample:
            upsampler = torch.hub.load("valeoai/NAF", "naf", pretrained=True)
            self.upsampler = upsampler.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.patch_size = self.encoder.patch_size
        self.patch_number = self.cfg.size // self.patch_size
        self.proj_grid = ProjGrid(grid_resolution=self.cfg.grid_resolution)

        # Fixed transformation matrix
        self.register_buffer("fix_transform_matrix", torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))

    def forward(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor = None,
        camera_angle_x: torch.Tensor = None,
        distance: torch.Tensor = None,
        mesh_scale: torch.Tensor = None,
        transform_matrix: torch.Tensor = None,
        is_training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            image: [B, num_views, C, H, W]
            camera_angle_x: [B, num_views]
            distance: [B, num_views]
            mesh_scale: [B]
            transform_matrix: [B, num_views, 4, 4]

        Returns:
            z_global: [B, num_global, C]
            z: [B, grid_resolution^3, C]
        """
        B, num_views, C, H, W = image.shape
        image = image.reshape(B * num_views, C, H, W)
        image = self.transform(image)

        with torch.no_grad():
            z = self.encoder(image, is_training=True)['x_prenorm']
            z = F.layer_norm(z, z.shape[-1:])
            z_clstoken = z[:, 0:1]
            z_regtokens = z[:, 1:self.encoder.num_register_tokens + 1]
            z_patchtokens = z[:, 1 + self.encoder.num_register_tokens:]
            z_patchtokens = z_patchtokens.reshape(
                z_patchtokens.shape[0],
                self.patch_number,
                self.patch_number,
                -1
            )

        # Compute relative transformation
        calc_mat, relative_transform = self.get_relative_transform(
            transform_matrix, distance
        )
        calc_mat = calc_mat.reshape(B * num_views, 4, 4)

        # Prepare parameters
        init_mesh_scale = mesh_scale[:, None].expand(B, num_views).reshape(B * num_views)
        camera_angle_x_flat = camera_angle_x.reshape(B * num_views)
        distance_flat = distance.reshape(B * num_views)

        # Accumulate per-view (avoid OOM)
        z_accumulated = None
        z_patchtokens_permuted = z_patchtokens.permute(0, 3, 1, 2) if self.use_upsample else None

        with torch.no_grad():
            for view_idx in range(num_views):
                indices = torch.arange(
                    view_idx, B * num_views, num_views, device=z_patchtokens.device
                )

                # Project current view
                z_view = self.proj_grid(
                    z_patchtokens[indices],
                    camera_angle_x_flat[indices],
                    distance_flat[indices],
                    init_mesh_scale[indices],
                    calc_mat[indices]
                )

                # Optional: upsample
                if self.use_upsample:
                    chunk_upsampled = self.upsampler(
                        image[indices],
                        z_patchtokens_permuted[indices],
                        output_size=(518, 518)
                    )
                    chunk_proj = self.proj_grid(
                        chunk_upsampled,
                        camera_angle_x_flat[indices],
                        distance_flat[indices],
                        init_mesh_scale[indices],
                        calc_mat[indices],
                        BHWC=False
                    )
                    z_view = z_view + chunk_proj
                    del chunk_upsampled, chunk_proj

                # Accumulate
                if z_accumulated is None:
                    z_accumulated = z_view.clone()
                else:
                    z_accumulated = z_accumulated + z_view
                del z_view

        if z_patchtokens_permuted is not None:
            del z_patchtokens_permuted

        # Average
        z = z_accumulated / num_views

        # Average global tokens
        z_global = torch.cat([z_clstoken, z_regtokens], dim=1)
        z_global = z_global.reshape(B, num_views, z_global.shape[-2], z_global.shape[-1])
        z_global = z_global.mean(dim=1)

        # Classifier-free guidance
        if is_training and random.random() < self.empty_embeds_ratio:
            z_global = z_global * 0
            z = z * 0

        return z_global, z

    def get_relative_transform(
        self,
        transform_matrix: torch.Tensor,
        distance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relative transformation matrix

        Args:
            transform_matrix: [B, num_views, 4, 4]
            distance: [B, num_views]

        Returns:
            calc_mat: [B, num_views, 4, 4]
            relative_transform: [B, num_views, 4, 4]
        """
        B, num_views, _, _ = transform_matrix.shape
        init_transform_matrix = transform_matrix[:, 0:1]

        fix_transform_matrix = self.fix_transform_matrix.unsqueeze(0).expand(B, -1, -1).clone()
        init_distance = distance[:, 0]
        fix_transform_matrix[:, 1, 3] = -init_distance
        fix_transform_matrix = fix_transform_matrix.unsqueeze(1)

        calc_mat, relative_transform = compute_calc_mat(
            init_transform_matrix, transform_matrix, fix_transform_matrix
        )
        return calc_mat, relative_transform
