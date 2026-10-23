from __future__ import annotations

import torch

from codex_zbot.perception.pointcloud import backproject_depth_world

from .config import OccupancyDriftConfig


def collect_observed_object_points(
    batch: dict[str, torch.Tensor],
    cfg: OccupancyDriftConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse foreground RGB-D surfaces from all observation frames."""
    all_points = []
    all_masks = []
    lower = torch.tensor(cfg.workspace_min, device=batch["depth"].device)
    upper = torch.tensor(cfg.workspace_max, device=batch["depth"].device)
    for frame in range(batch["depth"].shape[1]):
        dense, valid = backproject_depth_world(
            batch["depth"][:, frame].permute(0, 2, 3, 1),
            batch["intrinsics"],
            batch["pose"][:, frame],
            cfg.max_depth_m,
        )
        valid &= ((dense >= lower) & (dense <= upper)).all(dim=-1)
        # The collected semantic id 1 also covers the ground plane, so
        # ``semantic > 0`` is not a valid foreground mask.  Height filtering is
        # available both in simulation and on the real robot after extrinsic
        # calibration, and removes the dominant table/ground observations.
        valid &= dense[..., 2] > cfg.observed_min_height_m
        all_points.append(dense)
        all_masks.append(valid)
    dense = torch.cat(all_points, dim=1)
    valid = torch.cat(all_masks, dim=1)
    observed, masks = [], []
    for batch_id in range(dense.shape[0]):
        points = dense[batch_id, valid[batch_id]]
        count = min(points.shape[0], cfg.observed_points)
        output = torch.zeros(cfg.observed_points, 3, device=dense.device, dtype=dense.dtype)
        mask = torch.zeros(cfg.observed_points, device=dense.device, dtype=torch.bool)
        if count:
            ids = torch.randperm(points.shape[0], generator=generator, device="cpu")[:count].to(dense.device)
            output[:count] = points[ids]
            mask[:count] = True
        observed.append(output)
        masks.append(mask)
    return torch.stack(observed), torch.stack(masks)


def image_condition(batch: dict[str, torch.Tensor], max_depth_m: float) -> torch.Tensor:
    depth = batch["depth"]
    valid = torch.isfinite(depth) & (depth > 0) & (depth <= max_depth_m)
    normalized = torch.where(valid, depth / max_depth_m, torch.zeros_like(depth)).clamp(0, 1)
    return torch.cat([batch["rgb"], normalized, valid.float()], dim=2)
