"""Stable PPO feature contract."""
from __future__ import annotations

import torch


ACTOR_OBSERVATION_DIM = 173


def build_actor_observation(
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    camera_pose: torch.Tensor,
    reconstruction_latent: torch.Tensor,
    flow_stats: torch.Tensor,
    history_stats: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    expected = {
        "joint_position": 8,
        "joint_velocity": 8,
        "camera_pose": 7,
        "reconstruction_latent": 128,
        "flow_stats": 6,
        "history_stats": 8,
        "previous_action": 8,
    }
    tensors = {
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        "camera_pose": camera_pose,
        "reconstruction_latent": reconstruction_latent,
        "flow_stats": flow_stats,
        "history_stats": history_stats,
        "previous_action": previous_action,
    }
    for name, tensor in tensors.items():
        if tensor.ndim != 2 or tensor.shape[1] != expected[name]:
            raise ValueError(f"{name} must have shape (B,{expected[name]}), got {tuple(tensor.shape)}")
    observation = torch.cat(list(tensors.values()), dim=1)
    if observation.shape[1] != ACTOR_OBSERVATION_DIM:
        raise RuntimeError("Actor observation contract was changed without updating its declared dimension")
    return observation

