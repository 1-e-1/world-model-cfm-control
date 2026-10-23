from __future__ import annotations

import torch
import torch.nn as nn

from codex_zbot.config import WorldModelConfig
from codex_zbot.world_model.model import BeliefMetricHead, BeliefWorldModel

from .config import OccupancyDriftConfig
from .evidence import fixed_grid_centers


class V4BeliefEncoder(nn.Module):
    """Encode completed occupancy, tri-state memory and convergence statistics."""

    def __init__(self, drift_cfg: OccupancyDriftConfig, latent_dim: int = 128):
        super().__init__()
        centers = fixed_grid_centers(drift_cfg, torch.device("cpu"), torch.float32)
        self.register_buffer("grid_centers", centers)
        self.point_network = nn.Sequential(
            nn.Linear(3, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 192), nn.SiLU()
        )
        self.cell_network = nn.Sequential(
            nn.Linear(6, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 192), nn.SiLU()
        )
        self.stat_network = nn.Sequential(nn.Linear(9, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
        self.output = nn.Sequential(
            nn.Linear(192 * 4 + 96, 256), nn.SiLU(), nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim)
        )

    def forward(
        self,
        position: torch.Tensor,
        tri_state_probability: torch.Tensor,
        policy_features: torch.Tensor,
    ) -> torch.Tensor:
        if position.ndim != 3 or position.shape[-1] != 3:
            raise ValueError("position must have shape [B,N,3]")
        if tri_state_probability.ndim != 3 or tri_state_probability.shape[-1] != 3:
            raise ValueError("tri_state_probability must have shape [B,C,3]")
        if policy_features.shape != (position.shape[0], 9):
            raise ValueError("policy_features must have shape [B,9]")
        point = self.point_network(position)
        centers = self.grid_centers.to(position).unsqueeze(0).expand(position.shape[0], -1, -1)
        cell = self.cell_network(torch.cat([centers, tri_state_probability], dim=-1))
        pooled = torch.cat(
            [point.mean(dim=1), point.amax(dim=1), cell.mean(dim=1), cell.amax(dim=1), self.stat_network(policy_features)],
            dim=1,
        )
        return self.output(pooled)


class V4BeliefLearningSystem(nn.Module):
    def __init__(self, drift_cfg: OccupancyDriftConfig, world_cfg: WorldModelConfig):
        super().__init__()
        self.encoder = V4BeliefEncoder(drift_cfg, world_cfg.latent_dim)
        self.dynamics = BeliefWorldModel(world_cfg)
        self.metric_head = BeliefMetricHead(world_cfg.latent_dim)
