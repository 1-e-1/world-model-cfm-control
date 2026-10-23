"""Stable observation contract for the V4 occupancy-belief policy."""
from __future__ import annotations

import torch


V4_RECONSTRUCTION_FEATURE_DIM = 9
V4_HISTORY_FEATURE_DIM = 8
V4_ACTOR_OBSERVATION_DIM = 176
V4_CRITIC_OBSERVATION_DIM = 183


def build_v4_actor_observation(
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    camera_pose: torch.Tensor,
    belief_latent: torch.Tensor,
    reconstruction_features: torch.Tensor,
    history_features: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Concatenate only quantities available to the deployed visual servo."""
    expected = {
        "joint_position": 8,
        "joint_velocity": 8,
        "camera_pose": 7,
        "belief_latent": 128,
        "reconstruction_features": V4_RECONSTRUCTION_FEATURE_DIM,
        "history_features": V4_HISTORY_FEATURE_DIM,
        "previous_action": 8,
    }
    tensors = {
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        "camera_pose": camera_pose,
        "belief_latent": belief_latent,
        "reconstruction_features": reconstruction_features,
        "history_features": history_features,
        "previous_action": previous_action,
    }
    batch_size = joint_position.shape[0]
    for name, tensor in tensors.items():
        if tensor.ndim != 2 or tensor.shape != (batch_size, expected[name]):
            raise ValueError(f"{name} must have shape ({batch_size},{expected[name]}), got {tuple(tensor.shape)}")
    observation = torch.cat(list(tensors.values()), dim=1)
    if observation.shape[1] != V4_ACTOR_OBSERVATION_DIM:
        raise RuntimeError("V4 actor observation contract and declared dimension disagree")
    return observation


def build_v4_critic_observation(actor: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
    """Append seven simulator-only training signals to the actor observation."""
    if actor.ndim != 2 or actor.shape[1] != V4_ACTOR_OBSERVATION_DIM:
        raise ValueError(f"actor must have shape (B,{V4_ACTOR_OBSERVATION_DIM})")
    if privileged.shape != (actor.shape[0], 7):
        raise ValueError("privileged must have shape (B,7)")
    state = torch.cat([actor, privileged], dim=1)
    if state.shape[1] != V4_CRITIC_OBSERVATION_DIM:
        raise RuntimeError("V4 critic observation contract and declared dimension disagree")
    return state
