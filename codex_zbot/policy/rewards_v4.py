"""Reward terms for V4 occupancy completion and active camera control."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from codex_zbot.config import RewardConfig


@dataclass
class V4RewardTerms:
    total: torch.Tensor
    coverage: torch.Tensor
    reconstruction: torch.Tensor
    uncertainty: torch.Tensor
    unknown: torch.Tensor
    novelty: torch.Tensor
    visibility: torch.Tensor
    convergence: torch.Tensor
    proximity: torch.Tensor
    safety: torch.Tensor


def camera_to_oriented_boxes_clearance(
    camera_position: torch.Tensor,
    object_centers: torch.Tensor,
    object_sizes: torch.Tensor,
    object_yaws: torch.Tensor,
    object_active: torch.Tensor,
) -> torch.Tensor:
    """Return signed camera-to-nearest-OBB surface clearance for each environment."""
    delta = camera_position[:, None, :] - object_centers
    cosine = torch.cos(object_yaws)
    sine = torch.sin(object_yaws)
    local_x = cosine * delta[..., 0] + sine * delta[..., 1]
    local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    local = torch.stack((local_x, local_y, delta[..., 2]), dim=-1)
    if object_sizes.ndim == 2:
        object_sizes = object_sizes.unsqueeze(0)
    offset = local.abs() - 0.5 * object_sizes
    outside = offset.clamp_min(0.0).norm(dim=-1)
    inside = offset.amax(dim=-1).clamp_max(0.0)
    signed_distance = outside + inside
    signed_distance = torch.where(object_active, signed_distance, torch.full_like(signed_distance, torch.inf))
    return signed_distance.amin(dim=1)


def normalized_v4_active_vision_reward(
    cfg: RewardConfig,
    coverage_gain: torch.Tensor,
    chamfer_gain: torch.Tensor,
    uncertainty_gain: torch.Tensor,
    unknown_reduction: torch.Tensor,
    novelty: torch.Tensor,
    visibility: torch.Tensor,
    centroid_shift_previous: torch.Tensor,
    centroid_shift_current: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    joint_velocity: torch.Tensor,
    joint_margin_violation: torch.Tensor,
    collision: torch.Tensor,
    camera_proximity: torch.Tensor,
) -> V4RewardTerms:
    """Normalize information gains and penalize unsafe or ineffective motion.

    ``centroid_shift`` is a reconstruction convergence signal.  It is not
    interpreted as physical point velocity.
    """
    coverage = (coverage_gain / cfg.coverage_reference_gain).clamp(-1.0, 1.0)
    reconstruction = (chamfer_gain / cfg.chamfer_reference_gain_m).clamp(-1.0, 1.0)
    uncertainty = (uncertainty_gain / cfg.uncertainty_reference_gain).clamp(-1.0, 1.0)
    unknown = (unknown_reduction / cfg.unknown_reference_gain).clamp(-1.0, 1.0)
    novelty = novelty.clamp(0.0, 1.0)
    visibility = (visibility / cfg.visibility_reference).clamp(0.0, 1.0)
    information_gate = ((coverage > 0.0) | (reconstruction > 0.0) | (unknown > 0.0)).float()
    convergence = information_gate * (
        (centroid_shift_previous - centroid_shift_current) / cfg.centroid_shift_reference_m
    ).clamp(-1.0, 1.0)
    proximity = camera_proximity.clamp(0.0, 1.0)
    safety = (
        cfg.action_l2 * action.square().mean(dim=1)
        + cfg.action_rate * (action - previous_action).square().mean(dim=1)
        + cfg.joint_velocity * joint_velocity.square().mean(dim=1)
        + cfg.joint_margin * joint_margin_violation.float()
        + cfg.collision * collision.float()
        + cfg.camera_proximity * proximity
    )
    total = (
        cfg.coverage_gain * coverage
        + cfg.reconstruction_gain * reconstruction
        + cfg.uncertainty_gain * uncertainty
        + cfg.unknown_reduction * unknown
        + cfg.novel_view * novelty
        + cfg.visibility * visibility
        + cfg.flow_convergence * convergence
        + safety
    )
    return V4RewardTerms(
        total, coverage, reconstruction, uncertainty, unknown, novelty, visibility, convergence, proximity, safety
    )
