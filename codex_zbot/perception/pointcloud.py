"""Deterministic traditional point-cloud construction from RGB-D observations.

Camera poses use IsaacLab's ``quat_w_world`` convention: local +X is forward and local +Z is up.
The implementation never reads GT geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from codex_zbot.config import PerceptionConfig


def renderer_intrinsics_to_pixel_indices(intrinsics: torch.Tensor) -> torch.Tensor:
    """Isaac raster pixel centers are u+.5,v+.5; our arrays index u,v.

    Apply ONLY to native renderer readback, never to calibrated/stored K that
    already uses integer pixel indices. Verified on independent plane poses.
    """
    matrix=intrinsics.clone()
    matrix[...,0,2]-=.5
    matrix[...,1,2]-=.5
    return matrix


@dataclass
class FirstFramePointCloud:
    particles: torch.Tensor
    velocity: torch.Tensor
    confidence: torch.Tensor
    particle_type: torch.Tensor
    observed_points: torch.Tensor
    observed_mask: torch.Tensor
    valid_depth_ratio: torch.Tensor


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert normalized wxyz quaternions to rotation matrices."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def backproject_depth_world(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_world: torch.Tensor,
    max_depth_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project planar depth to world points.

    Args:
        depth: ``(B,H,W)`` or ``(B,H,W,1)`` planar depth in metres.
        intrinsics: ``(B,3,3)`` or ``(3,3)``.
        pose_world: ``(B,7)`` as xyz + wxyz, in IsaacLab world-camera convention.
    Returns:
        Dense points ``(B,H*W,3)`` and validity mask ``(B,H*W)``.
    """
    if depth.ndim == 4:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError("depth must have shape (B,H,W) or (B,H,W,1)")
    batch, height, width = depth.shape
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(batch, -1, -1)
    valid = torch.isfinite(depth) & (depth > 0.0) & (depth <= max_depth_m)
    distance = torch.where(valid, depth, torch.zeros_like(depth))

    u = torch.arange(width, device=depth.device, dtype=depth.dtype)
    v = torch.arange(height, device=depth.device, dtype=depth.dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    fx = intrinsics[:, 0, 0].clamp_min(1.0e-6)
    fy = intrinsics[:, 1, 1].clamp_min(1.0e-6)
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    dx = (uu.unsqueeze(0) - cx[:, None, None]) / fx[:, None, None]
    dy = (vv.unsqueeze(0) - cy[:, None, None]) / fy[:, None, None]

    rotation = quaternion_to_matrix(pose_world[:, 3:])
    forward = rotation[:, :, 0]
    left = rotation[:, :, 1]
    up = rotation[:, :, 2]
    position = pose_world[:, :3]
    points = (
        position[:, None, None]
        + distance[..., None] * forward[:, None, None]
        - (distance * dx)[..., None] * left[:, None, None]
        - (distance * dy)[..., None] * up[:, None, None]
    )
    return points.reshape(batch, -1, 3), valid.reshape(batch, -1)


def _sample_fixed(points: torch.Tensor, count: int, generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample/pad one point set and return a validity mask."""
    device = points.device
    n = points.shape[0]
    if n == 0:
        return torch.zeros(count, 3, device=device, dtype=points.dtype), torch.zeros(count, device=device, dtype=torch.bool)
    if n >= count:
        ids = torch.randperm(n, generator=generator, device="cpu")[:count].to(device)
        return points[ids], torch.ones(count, device=device, dtype=torch.bool)
    output = torch.zeros(count, 3, device=device, dtype=points.dtype)
    output[:n] = points
    return output, torch.arange(count, device=device) < n


def _voxel_downsample(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    if points.numel() == 0 or voxel_size <= 0:
        return points
    keys = torch.floor(points / voxel_size).to(torch.int64)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    count = int(inverse.amax().item()) + 1
    sums = torch.zeros(count, 3, device=points.device, dtype=points.dtype)
    sums.index_add_(0, inverse, points)
    counts = torch.bincount(inverse, minlength=count).to(points.dtype).unsqueeze(1)
    return sums / counts.clamp_min(1.0)


def _radius_filter(points: torch.Tensor, radius: float, min_neighbors: int, max_candidates: int = 4096) -> torch.Tensor:
    if points.shape[0] <= min_neighbors or radius <= 0:
        return points
    if points.shape[0] > max_candidates:
        # Deterministic stride reduction bounds the O(N^2) traditional filter.
        step = max(1, points.shape[0] // max_candidates)
        points = points[::step][:max_candidates]
    neighbor_count = (torch.cdist(points, points) <= radius).sum(dim=1) - 1
    kept = points[neighbor_count >= min_neighbors]
    return kept if kept.shape[0] > 0 else points


def build_first_frame_cloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_world: torch.Tensor,
    cfg: PerceptionConfig,
    semantic: torch.Tensor | None = None,
    foreground_ids: set[int] | None = None,
    generator: torch.Generator | None = None,
) -> FirstFramePointCloud:
    """Build observed points and a fixed-size particle cloud from the first camera frame."""
    dense, valid = backproject_depth_world(depth, intrinsics, pose_world, cfg.max_depth_m)
    if semantic is not None and foreground_ids:
        if semantic.ndim == 4:
            semantic = semantic[..., 0]
        sem = semantic.reshape(semantic.shape[0], -1)
        foreground = torch.zeros_like(valid)
        for class_id in foreground_ids:
            foreground |= sem == int(class_id)
        valid &= foreground

    lower = torch.tensor(cfg.workspace_min, device=dense.device, dtype=dense.dtype)
    upper = torch.tensor(cfg.workspace_max, device=dense.device, dtype=dense.dtype)
    valid &= ((dense >= lower) & (dense <= upper)).all(dim=-1)
    valid_ratio = valid.float().mean(dim=1)

    observed_batch = []
    observed_mask_batch = []
    particles_batch = []
    confidence_batch = []
    type_batch = []
    latent_count = min(cfg.num_points - 1, round(cfg.num_points * cfg.latent_particle_ratio))
    measured_count = cfg.num_points - latent_count
    for batch_id in range(dense.shape[0]):
        points = dense[batch_id, valid[batch_id]]
        points = _voxel_downsample(points, cfg.voxel_size_m)
        points = _radius_filter(points, cfg.outlier_radius_m, cfg.outlier_min_neighbors)
        observed, observed_mask = _sample_fixed(points, cfg.observed_points, generator)
        measured, measured_mask = _sample_fixed(points, measured_count, generator)

        latent = torch.rand(latent_count, 3, generator=generator, device="cpu", dtype=dense.dtype).to(dense.device)
        latent = lower + latent * (upper - lower)
        particles = torch.cat([measured, latent], dim=0)
        measured_confidence = measured_mask.to(dense.dtype)
        confidence = torch.cat([measured_confidence, torch.zeros(latent_count, device=dense.device)], dim=0)
        particle_type = torch.cat(
            [torch.zeros(measured_count, device=dense.device), torch.ones(latent_count, device=dense.device)], dim=0
        )
        observed_batch.append(observed)
        observed_mask_batch.append(observed_mask)
        particles_batch.append(particles)
        confidence_batch.append(confidence.unsqueeze(-1))
        type_batch.append(particle_type.unsqueeze(-1))

    particles = torch.stack(particles_batch)
    return FirstFramePointCloud(
        particles=particles,
        velocity=torch.zeros_like(particles),
        confidence=torch.stack(confidence_batch),
        particle_type=torch.stack(type_batch),
        observed_points=torch.stack(observed_batch),
        observed_mask=torch.stack(observed_mask_batch),
        valid_depth_ratio=valid_ratio,
    )


def prepare_image_condition(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    image_size: int,
    max_depth_m: float,
) -> torch.Tensor:
    """Return RGB + normalized depth + valid mask as ``(B,5,S,S)``."""
    if rgb.ndim != 4:
        raise ValueError("rgb must have four dimensions")
    if rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2)
    rgb = rgb.float() / 255.0 if rgb.dtype == torch.uint8 or rgb.amax() > 1.0 else rgb.float()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.permute(0, 3, 1, 2)
    elif depth.ndim == 3:
        depth = depth.unsqueeze(1)
    valid = torch.isfinite(depth) & (depth > 0.0) & (depth <= max_depth_m)
    normalized_depth = torch.where(valid, depth / max_depth_m, torch.zeros_like(depth)).clamp(0.0, 1.0)
    condition = torch.cat([rgb, normalized_depth, valid.float()], dim=1)
    return F.interpolate(condition, (image_size, image_size), mode="bilinear", align_corners=False)
