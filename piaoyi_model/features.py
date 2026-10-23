from __future__ import annotations

import itertools

import torch
import torch.nn as nn

from .config import OccupancyDriftConfig


def soft_voxel_occupancy(points: torch.Tensor, resolution: int, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Differentiable trilinear splatting into an occupancy grid."""
    batch, samples, count, _ = points.shape
    normalized = ((points - lower) / (upper - lower).clamp_min(1.0e-6)).clamp(0.0, 1.0 - 1.0e-6)
    coordinates = normalized * (resolution - 1)
    base = coordinates.floor().long()
    fraction = coordinates - base.to(coordinates.dtype)
    flat_batch = batch * samples
    base = base.reshape(flat_batch, count, 3)
    fraction = fraction.reshape(flat_batch, count, 3)
    occupancy = torch.zeros(flat_batch, resolution**3, device=points.device, dtype=points.dtype)
    for offset_tuple in itertools.product((0, 1), repeat=3):
        offset = torch.tensor(offset_tuple, device=points.device)
        index_3d = (base + offset).clamp(0, resolution - 1)
        weight = torch.ones(flat_batch, count, device=points.device, dtype=points.dtype)
        for axis, bit in enumerate(offset_tuple):
            weight = weight * (fraction[..., axis] if bit else 1.0 - fraction[..., axis])
        index = index_3d[..., 0] * resolution * resolution + index_3d[..., 1] * resolution + index_3d[..., 2]
        occupancy.scatter_add_(1, index, weight)
    occupancy = 1.0 - torch.exp(-occupancy)
    return occupancy.reshape(batch, samples, -1)


class MultiScaleOccupancyFeatures(nn.Module):
    """Fixed permutation-invariant features used only during drifting training."""

    def __init__(self, cfg: OccupancyDriftConfig):
        super().__init__()
        self.resolutions = tuple(cfg.voxel_resolutions)
        self.register_buffer("lower", torch.tensor(cfg.workspace_min).view(1, 1, 1, 3))
        self.register_buffer("upper", torch.tensor(cfg.workspace_max).view(1, 1, 1, 3))

    def forward(self, points: torch.Tensor) -> list[torch.Tensor]:
        features = [soft_voxel_occupancy(points, resolution, self.lower, self.upper) for resolution in self.resolutions]
        mean = points.mean(dim=2)
        centered = points - mean.unsqueeze(2)
        std = centered.square().mean(dim=2).add(1.0e-8).sqrt()
        covariance = torch.einsum("bknc,bknd->bkcd", centered, centered) / max(1, points.shape[2] - 1)
        features.append(torch.cat([mean, std, covariance.flatten(2)], dim=-1))
        return features
