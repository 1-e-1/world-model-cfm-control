"""Point-cloud and active-vision metrics."""
from __future__ import annotations

import torch


def chamfer_distance(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Symmetric Chamfer distance for batched point clouds without a full NxM allocation."""
    if a.ndim != 3 or b.ndim != 3 or a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("a and b must have shape (B, N, 3) and (B, M, 3)")
    mins_a = []
    for start in range(0, a.shape[1], chunk_size):
        mins_a.append(torch.cdist(a[:, start : start + chunk_size], b).amin(dim=2))
    mins_b = []
    for start in range(0, b.shape[1], chunk_size):
        mins_b.append(torch.cdist(b[:, start : start + chunk_size], a).amin(dim=2))
    return torch.cat(mins_a, dim=1).mean(dim=1) + torch.cat(mins_b, dim=1).mean(dim=1)


def coverage_at_threshold(pred: torch.Tensor, target: torch.Tensor, threshold_m: float) -> torch.Tensor:
    distances = torch.cdist(target, pred).amin(dim=2)
    return (distances < threshold_m).float().mean(dim=1)


def flow_statistics(position: torch.Tensor, previous: torch.Tensor, velocity: torch.Tensor, acceleration: torch.Tensor) -> torch.Tensor:
    speed = velocity.norm(dim=-1)
    accel = acceleration.norm(dim=-1)
    displacement = (position - previous).norm(dim=-1)
    p90 = torch.quantile(speed, 0.9, dim=1)
    converged = (speed < 0.01).float().mean(dim=1)
    return torch.stack(
        [speed.mean(1), p90, speed.amax(1), accel.mean(1), displacement.mean(1), converged], dim=1
    )

