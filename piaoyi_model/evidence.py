from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from codex_zbot.perception.pointcloud import backproject_depth_world

from .config import OccupancyDriftConfig


@dataclass
class RayEvidence:
    surface_points: torch.Tensor  # (B,T,R*R,3)
    surface_mask: torch.Tensor  # (B,T,R*R)
    free_points: torch.Tensor  # (B,T,R*R*S,3)
    free_mask: torch.Tensor  # (B,T,R*R*S)


def build_ray_evidence(batch: dict[str, torch.Tensor], cfg: OccupancyDriftConfig) -> RayEvidence:
    """Lift low-resolution depth rays and mark pre-surface samples as free.

    This operation uses camera calibration only.  Object boxes and semantic GT
    are deliberately not read, so the same evidence is available at inference.
    """
    depth = batch["depth"]
    batch_size, frames, _, source_height, source_width = depth.shape
    grid = cfg.ray_grid_size
    small = F.interpolate(
        depth.reshape(batch_size * frames, 1, source_height, source_width),
        (grid, grid),
        mode="nearest",
    )[:, 0]
    intrinsics = batch["intrinsics"].clone()
    intrinsics[:, 0, [0, 2]] *= grid / source_width
    intrinsics[:, 1, [1, 2]] *= grid / source_height
    intrinsics = intrinsics[:, None].expand(-1, frames, -1, -1).reshape(batch_size * frames, 3, 3)
    pose = batch["pose"].reshape(batch_size * frames, 7)
    surface, valid = backproject_depth_world(small, intrinsics, pose, cfg.max_depth_m)
    lower = torch.as_tensor(cfg.workspace_min, device=depth.device, dtype=depth.dtype)
    upper = torch.as_tensor(cfg.workspace_max, device=depth.device, dtype=depth.dtype)
    valid &= ((surface >= lower) & (surface <= upper)).all(dim=-1)
    valid &= surface[..., 2] > cfg.observed_min_height_m

    camera = pose[:, None, :3]
    fractions = torch.linspace(
        cfg.free_ray_end_fraction / (cfg.free_samples_per_ray + 1),
        cfg.free_ray_end_fraction,
        cfg.free_samples_per_ray,
        device=depth.device,
        dtype=depth.dtype,
    )
    free = camera[:, :, None, :] + fractions[None, None, :, None] * (surface[:, :, None, :] - camera[:, :, None, :])
    free_valid = valid[:, :, None].expand(-1, -1, cfg.free_samples_per_ray).clone()
    free_valid &= ((free >= lower) & (free <= upper)).all(dim=-1)
    free = free.reshape(batch_size, frames, grid * grid * cfg.free_samples_per_ray, 3)
    free_valid = free_valid.reshape(batch_size, frames, grid * grid * cfg.free_samples_per_ray)
    return RayEvidence(
        surface.reshape(batch_size, frames, grid * grid, 3),
        valid.reshape(batch_size, frames, grid * grid),
        free,
        free_valid,
    )


def evidence_grid(
    evidence: RayEvidence,
    cfg: OccupancyDriftConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-cell [surface, free] evidence and corresponding masks."""
    resolution = cfg.memory_resolution
    lower = torch.as_tensor(cfg.workspace_min, device=evidence.surface_points.device, dtype=evidence.surface_points.dtype)
    upper = torch.as_tensor(cfg.workspace_max, device=lower.device, dtype=lower.dtype)
    channels = []
    masks = []
    for points, mask in (
        (evidence.surface_points.flatten(1, 2), evidence.surface_mask.flatten(1, 2)),
        (evidence.free_points.flatten(1, 2), evidence.free_mask.flatten(1, 2)),
    ):
        normalized = ((points - lower) / (upper - lower).clamp_min(1.0e-6)).clamp(0, 1 - 1.0e-6)
        xyz = (normalized * resolution).long().clamp(0, resolution - 1)
        ids = xyz[..., 0] * resolution * resolution + xyz[..., 1] * resolution + xyz[..., 2]
        counts = torch.zeros(points.shape[0], resolution**3, device=points.device, dtype=points.dtype)
        counts.scatter_add_(1, ids, mask.to(points.dtype))
        channels.append(torch.log1p(counts))
        masks.append(counts > 0)
    return torch.stack(channels, dim=-1), masks[0], masks[1]


def fixed_grid_centers(cfg: OccupancyDriftConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    resolution = cfg.memory_resolution
    lower = torch.as_tensor(cfg.workspace_min, device=device, dtype=dtype)
    upper = torch.as_tensor(cfg.workspace_max, device=device, dtype=dtype)
    axes = [torch.linspace(lower[i], upper[i], resolution + 1, device=device, dtype=dtype) for i in range(3)]
    axes = [0.5 * (axis[:-1] + axis[1:]) for axis in axes]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
