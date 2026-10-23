"""Compact reconstruction belief and action-conditioned dynamics."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from codex_zbot.config import WorldModelConfig
from codex_zbot.utils.metrics import flow_statistics


class ReconstructionEncoder(nn.Module):
    """Encode point position, flow velocity and confidence to a compact latent."""

    def __init__(self, latent_dim: int = 128, point_dim: int = 256):
        super().__init__()
        self.point_network = nn.Sequential(
            nn.Linear(3 + 3 + 1 + 1, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, point_dim),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(point_dim * 2 + 6, 256), nn.SiLU(), nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim)
        )

    def forward(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        confidence: torch.Tensor,
        particle_type: torch.Tensor,
        previous_position: torch.Tensor | None = None,
        acceleration: torch.Tensor | None = None,
    ) -> torch.Tensor:
        previous_position = position if previous_position is None else previous_position
        acceleration = torch.zeros_like(position) if acceleration is None else acceleration
        feature = self.point_network(torch.cat([position, velocity, confidence, particle_type], dim=-1))
        pooled = torch.cat([feature.mean(dim=1), feature.amax(dim=1)], dim=1)
        stats = flow_statistics(position, previous_position, velocity, acceleration)
        return self.output(torch.cat([pooled, stats], dim=1))


class BeliefMetricHead(nn.Module):
    """Prevent latent collapse by decoding measurable reconstruction statistics."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


@dataclass
class BeliefPrediction:
    next_latent: torch.Tensor
    coverage_gain: torch.Tensor
    reconstruction_gain: torch.Tensor
    uncertainty: torch.Tensor
    collision_logit: torch.Tensor
    hidden: torch.Tensor


class BeliefWorldModel(nn.Module):
    """Predict action-conditioned belief transition and task-relevant outcomes."""

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.latent_dim + cfg.robot_state_dim + cfg.action_dim
        self.input = nn.Sequential(nn.Linear(input_dim, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.recurrent = nn.GRUCell(cfg.hidden_dim, cfg.recurrent_dim)
        head_input = cfg.recurrent_dim + cfg.latent_dim
        self.latent_head = nn.Sequential(
            nn.Linear(head_input, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        )
        self.scalar_head = nn.Sequential(
            nn.Linear(head_input, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, 4)
        )

    def initial_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(batch_size, self.cfg.recurrent_dim, device=device)

    def forward(
        self,
        latent: torch.Tensor,
        robot_state: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> BeliefPrediction:
        if hidden is None:
            hidden = self.initial_hidden(latent.shape[0], latent.device)
        recurrent_input = self.input(torch.cat([latent, robot_state, action], dim=-1))
        hidden = self.recurrent(recurrent_input, hidden)
        head_input = torch.cat([hidden, latent], dim=-1)
        next_latent = F.layer_norm(latent + self.latent_head(head_input), (latent.shape[-1],))
        scalars = self.scalar_head(head_input)
        return BeliefPrediction(
            next_latent=next_latent,
            coverage_gain=scalars[:, 0],
            reconstruction_gain=scalars[:, 1],
            uncertainty=F.softplus(scalars[:, 2]),
            collision_logit=scalars[:, 3],
            hidden=hidden,
        )

    def rollout(
        self,
        latent: torch.Tensor,
        robot_states: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> list[BeliefPrediction]:
        predictions = []
        for step in range(actions.shape[1]):
            prediction = self(latent, robot_states[:, step], actions[:, step], hidden)
            predictions.append(prediction)
            latent, hidden = prediction.next_latent, prediction.hidden
        return predictions


class BeliefLearningSystem(nn.Module):
    """Checkpointable bundle used while learning the belief representation and dynamics."""

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.encoder = ReconstructionEncoder(cfg.latent_dim)
        self.dynamics = BeliefWorldModel(cfg)
        self.metric_head = BeliefMetricHead(cfg.latent_dim)


def world_model_loss(
    predictions: list[BeliefPrediction],
    target_latents: torch.Tensor,
    coverage_gain: torch.Tensor,
    reconstruction_gain: torch.Tensor,
    collision: torch.Tensor,
    discount: float = 0.9,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = {"latent": 0.0, "coverage": 0.0, "reconstruction": 0.0, "uncertainty": 0.0, "collision": 0.0}
    normalizer = 0.0
    for step, prediction in enumerate(predictions):
        weight = discount**step
        error = prediction.next_latent - target_latents[:, step]
        uncertainty = prediction.uncertainty.clamp_min(1.0e-4)
        latent_nll = 0.5 * (error.square().mean(dim=1) / uncertainty)
        uncertainty_nll = 0.5 * uncertainty.log()
        losses["latent"] = losses["latent"] + weight * latent_nll.mean()
        losses["uncertainty"] = losses["uncertainty"] + weight * uncertainty_nll.mean()
        losses["coverage"] = losses["coverage"] + weight * F.huber_loss(
            prediction.coverage_gain, coverage_gain[:, step]
        )
        losses["reconstruction"] = losses["reconstruction"] + weight * F.huber_loss(
            prediction.reconstruction_gain, reconstruction_gain[:, step]
        )
        losses["collision"] = losses["collision"] + weight * F.binary_cross_entropy_with_logits(
            prediction.collision_logit, collision[:, step].float()
        )
        normalizer += weight
    tensor_losses = {key: value / normalizer for key, value in losses.items()}
    total = (
        tensor_losses["latent"]
        + tensor_losses["uncertainty"]
        + tensor_losses["coverage"]
        + tensor_losses["reconstruction"]
        + tensor_losses["collision"]
    )
    return total, tensor_losses
