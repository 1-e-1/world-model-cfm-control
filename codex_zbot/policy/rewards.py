"""Depth-aware coverage and normalized active-vision reward."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from codex_zbot.config import RewardConfig
from codex_zbot.perception.pointcloud import quaternion_to_matrix


def depth_consistent_visibility(
    gt_points_world: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose_world: torch.Tensor,
    tolerance_m: float = 0.025,
    max_depth_m: float = 2.0,
) -> torch.Tensor:
    """Return GT visibility using projection plus measured depth occlusion.

    Camera orientation follows IsaacLab world convention: local +X forward, +Y left, +Z up.
    """
    if depth.ndim == 4:
        depth = depth[..., 0]
    batch, height, width = depth.shape
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(batch, -1, -1)
    relative_world = gt_points_world - camera_pose_world[:, None, :3]
    rotation = quaternion_to_matrix(camera_pose_world[:, 3:])
    local = torch.bmm(relative_world, rotation)
    forward = local[..., 0]
    u = intrinsics[:, None, 0, 2] - intrinsics[:, None, 0, 0] * local[..., 1] / forward.clamp_min(1.0e-6)
    v = intrinsics[:, None, 1, 2] - intrinsics[:, None, 1, 1] * local[..., 2] / forward.clamp_min(1.0e-6)
    ui = u.round().long()
    vi = v.round().long()
    inside = (forward > 0.0) & (forward <= max_depth_m) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    linear = (vi.clamp(0, height - 1) * width + ui.clamp(0, width - 1)).long()
    measured = depth.reshape(batch, -1).gather(1, linear)
    valid_measurement = torch.isfinite(measured) & (measured > 0.0) & (measured <= max_depth_m)
    return inside & valid_measurement & ((measured - forward).abs() <= tolerance_m)


def points_inside_image(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose_world: torch.Tensor,
    image_height: int,
    image_width: int,
    margin_px: float = 0.0,
    max_depth_m: float = 2.0,
) -> torch.Tensor:
    """Return geometric FOV containment with an optional image-border margin."""
    batch = points_world.shape[0]
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(batch, -1, -1)
    relative_world = points_world - camera_pose_world[:, None, :3]
    rotation = quaternion_to_matrix(camera_pose_world[:, 3:])
    local = torch.bmm(relative_world, rotation)
    forward = local[..., 0]
    u = intrinsics[:, None, 0, 2] - intrinsics[:, None, 0, 0] * local[..., 1] / forward.clamp_min(1.0e-6)
    v = intrinsics[:, None, 1, 2] - intrinsics[:, None, 1, 1] * local[..., 2] / forward.clamp_min(1.0e-6)
    return (
        (forward > 0.0)
        & (forward <= max_depth_m)
        & (u >= margin_px)
        & (u < image_width - margin_px)
        & (v >= margin_px)
        & (v < image_height - margin_px)
    )


def visibility_by_object(visible_points: torch.Tensor, points_per_object: int) -> torch.Tensor:
    """Return one visible-surface ratio per equal contiguous object point block."""
    if points_per_object <= 0:
        raise ValueError("points_per_object must be positive")
    if visible_points.shape[-1] % points_per_object != 0:
        raise ValueError("visibility point count must be divisible by points_per_object")
    object_count = visible_points.shape[-1] // points_per_object
    return visible_points.reshape(*visible_points.shape[:-1], object_count, points_per_object).float().mean(dim=-1)


@dataclass
class RewardTerms:
    total: torch.Tensor
    coverage: torch.Tensor
    reconstruction: torch.Tensor
    uncertainty: torch.Tensor
    novelty: torch.Tensor
    visibility: torch.Tensor
    flow: torch.Tensor
    safety: torch.Tensor


def normalized_active_vision_reward(
    cfg: RewardConfig,
    coverage_gain: torch.Tensor,
    chamfer_gain: torch.Tensor,
    uncertainty_gain: torch.Tensor,
    novelty: torch.Tensor,
    visibility: torch.Tensor,
    flow_speed_previous: torch.Tensor,
    flow_speed_current: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    joint_velocity: torch.Tensor,
    joint_margin_violation: torch.Tensor,
    collision: torch.Tensor,
) -> RewardTerms:
    coverage = (coverage_gain / cfg.coverage_reference_gain).clamp(-1.0, 1.0)
    reconstruction = (chamfer_gain / cfg.chamfer_reference_gain_m).clamp(-1.0, 1.0)
    uncertainty = uncertainty_gain.clamp(-1.0, 1.0)
    novelty = novelty.clamp(0.0, 1.0)
    visibility = visibility.clamp(0.0, 1.0)
    information_gate = ((coverage > 0.0) | (reconstruction > 0.0)).float()
    flow = information_gate * ((flow_speed_previous - flow_speed_current) / 0.01).clamp(-1.0, 1.0)
    safety = (
        cfg.action_l2 * action.square().mean(dim=1)
        + cfg.action_rate * (action - previous_action).square().mean(dim=1)
        + cfg.joint_velocity * joint_velocity.square().mean(dim=1)
        + cfg.joint_margin * joint_margin_violation.float()
        + cfg.collision * collision.float()
    )
    total = (
        cfg.coverage_gain * coverage
        + cfg.reconstruction_gain * reconstruction
        + cfg.uncertainty_gain * uncertainty
        + cfg.novel_view * novelty
        + cfg.visibility * visibility
        + cfg.flow_convergence * flow
        + safety
    )
    return RewardTerms(total, coverage, reconstruction, uncertainty, novelty, visibility, flow, safety)
