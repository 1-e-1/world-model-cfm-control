"""Second-order rest-to-rest CFM losses."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from codex_zbot.config import CFMLossConfig
from codex_zbot.utils.metrics import chamfer_distance

from .model import CFMCondition, SecondOrderCFM


@dataclass
class CFMLossOutput:
    total: torch.Tensor
    acceleration: torch.Tensor
    terminal_position: torch.Tensor
    terminal_velocity: torch.Tensor
    terminal_acceleration: torch.Tensor
    observation_consistency: torch.Tensor
    repulsion: torch.Tensor
    confidence: torch.Tensor
    mean_terminal_speed: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            key: float(value.detach().item())
            for key, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }


def quintic_path(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Position, velocity and acceleration coefficients with zero endpoint velocity/acceleration."""
    t2 = time * time
    t3 = t2 * time
    t4 = t3 * time
    t5 = t4 * time
    position = 10 * t3 - 15 * t4 + 6 * t5
    velocity = 30 * t2 - 60 * t3 + 30 * t4
    acceleration = 60 * time - 180 * t2 + 120 * t3
    return position, velocity, acceleration


@torch.no_grad()
def sinkhorn_barycentric_match(
    source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 0.03,
    iterations: int = 20,
) -> torch.Tensor:
    """Balanced entropic OT barycentric target for equally weighted point sets."""
    cost = torch.cdist(source, target).square()
    log_kernel = -cost / max(epsilon, 1.0e-6)
    log_a = torch.full(
        source.shape[:2], -torch.log(torch.tensor(float(source.shape[1]), device=source.device)), device=source.device
    )
    log_b = torch.full(
        target.shape[:2], -torch.log(torch.tensor(float(target.shape[1]), device=target.device)), device=target.device
    )
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[:, None, :], dim=2)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, :, None], dim=1)
    plan = torch.exp(log_kernel + log_u[:, :, None] + log_v[:, None, :])
    return torch.bmm(plan, target) / plan.sum(dim=2, keepdim=True).clamp_min(1.0e-8)


def _subsample_pair(
    source: torch.Tensor,
    target: torch.Tensor,
    count: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = min(count, source.shape[1], target.shape[1])
    source_ids = torch.randperm(source.shape[1], generator=generator, device="cpu")[:count].to(source.device)
    target_ids = torch.randperm(target.shape[1], generator=generator, device="cpu")[:count].to(target.device)
    return source[:, source_ids], target[:, target_ids], source_ids


def _repulsion_loss(points: torch.Tensor, radius: float = 0.01, max_points: int = 512) -> torch.Tensor:
    if points.shape[1] > max_points:
        points = points[:, :max_points]
    distances = torch.cdist(points, points)
    diagonal = torch.eye(points.shape[1], device=points.device, dtype=torch.bool).unsqueeze(0)
    nearest = distances.masked_fill(diagonal, float("inf")).amin(dim=2)
    return F.relu(radius - nearest).square().mean()


def compute_cfm_loss(
    model: SecondOrderCFM,
    condition: CFMCondition,
    target: torch.Tensor,
    cfg: CFMLossConfig,
    generator: torch.Generator | None = None,
) -> CFMLossOutput:
    context = model.encode_condition(condition)
    source, target_sub, source_ids = _subsample_pair(
        condition.previous_points, target, cfg.match_points, generator
    )
    velocity_0 = condition.previous_velocity[:, source_ids]
    confidence = condition.confidence[:, source_ids]
    particle_type = condition.particle_type[:, source_ids]
    matched = sinkhorn_barycentric_match(source, target_sub, cfg.sinkhorn_epsilon, cfg.sinkhorn_iterations)
    displacement = matched - source

    batch = source.shape[0]
    time = torch.rand(batch, generator=generator, device="cpu", dtype=source.dtype).to(source.device)
    position_coefficient, velocity_coefficient, acceleration_coefficient = quintic_path(time)
    path_position = source + displacement * position_coefficient[:, None, None]
    path_velocity = velocity_0 + displacement * velocity_coefficient[:, None, None]
    target_acceleration = displacement * acceleration_coefficient[:, None, None]
    field_target = (
        target_acceleration
        if model.cfg.dynamics_order == 2
        else displacement * velocity_coefficient[:, None, None]
    )
    field_velocity = path_velocity if model.cfg.dynamics_order == 2 else torch.zeros_like(path_velocity)
    predicted_acceleration, _ = model.acceleration(
        path_position, field_velocity, confidence, particle_type, time, context
    )
    acceleration_loss = F.mse_loss(predicted_acceleration, field_target)

    terminal = model.integrate(
        source,
        velocity_0,
        confidence,
        particle_type,
        context,
        num_steps=cfg.terminal_steps,
    )
    terminal_position_loss = chamfer_distance(terminal.position, target_sub).mean()
    terminal_velocity_loss = terminal.velocity.square().mean()
    terminal_acceleration_loss = terminal.acceleration.square().mean()

    observed = condition.observed_points
    observed_mask = condition.observed_mask
    observed_distance = torch.cdist(observed, terminal.position).amin(dim=2)
    observation_loss = (observed_distance * observed_mask).sum() / observed_mask.sum().clamp_min(1)
    repulsion_loss = _repulsion_loss(terminal.position)
    confidence_target = (torch.cdist(terminal.position, target_sub).amin(dim=2) < 0.03).float().unsqueeze(-1)
    # The flow state exposes probabilities for downstream policy features.  Map
    # them back to logits here so the training loss remains CUDA autocast-safe.
    confidence_logit = torch.logit(terminal.confidence.float().clamp(1.0e-5, 1.0 - 1.0e-5))
    confidence_loss = F.binary_cross_entropy_with_logits(confidence_logit, confidence_target.float())

    total = (
        cfg.acceleration * acceleration_loss
        + cfg.terminal_position * terminal_position_loss
        + cfg.terminal_velocity * terminal_velocity_loss
        + cfg.terminal_acceleration * terminal_acceleration_loss
        + cfg.observation_consistency * observation_loss
        + cfg.repulsion * repulsion_loss
        + cfg.confidence * confidence_loss
    )
    return CFMLossOutput(
        total=total,
        acceleration=acceleration_loss,
        terminal_position=terminal_position_loss,
        terminal_velocity=terminal_velocity_loss,
        terminal_acceleration=terminal_acceleration_loss,
        observation_consistency=observation_loss,
        repulsion=repulsion_loss,
        confidence=confidence_loss,
        mean_terminal_speed=terminal.velocity.norm(dim=-1).mean(),
    )
