"""Sequence rollout utilities shared by training, evaluation and the RL environment."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from codex_zbot.config import CFMLossConfig, PerceptionConfig
from codex_zbot.perception.pointcloud import build_first_frame_cloud, prepare_image_condition

from .losses import CFMLossOutput, compute_cfm_loss
from .model import CFMCondition, FlowState, SecondOrderCFM


@dataclass
class SequenceRollout:
    states: list[FlowState]
    losses: list[CFMLossOutput]
    valid_depth_ratio: torch.Tensor
    particle_type: torch.Tensor


def _frame_frontend(
    batch: dict[str, torch.Tensor],
    frame: int,
    perception_cfg: PerceptionConfig,
    generator: torch.Generator | None,
):
    return build_first_frame_cloud(
        batch["depth"][:, frame].permute(0, 2, 3, 1),
        batch["intrinsics"],
        batch["pose"][:, frame],
        perception_cfg,
        generator=generator,
    )


def _anchor_prediction(
    state: FlowState,
    particle_type: torch.Tensor,
    observed_points: torch.Tensor,
    observed_mask: torch.Tensor,
) -> FlowState:
    """Replace the lowest-confidence particles with current measured points, without GT."""
    position = state.position.detach().clone()
    velocity = state.velocity.detach().clone()
    confidence = state.confidence.detach().clone()
    for batch_id in range(position.shape[0]):
        valid_observed = observed_points[batch_id, observed_mask[batch_id]]
        replace_count = min(valid_observed.shape[0], max(1, position.shape[1] // 4))
        if replace_count == 0:
            continue
        low_confidence = confidence[batch_id, :, 0].argsort()[:replace_count]
        position[batch_id, low_confidence] = valid_observed[:replace_count]
        velocity[batch_id, low_confidence] = 0.0
        confidence[batch_id, low_confidence] = 1.0
    return FlowState(position, velocity, confidence, state.acceleration.detach())


def rollout_sequence(
    model: SecondOrderCFM,
    batch: dict[str, torch.Tensor],
    perception_cfg: PerceptionConfig,
    loss_cfg: CFMLossConfig | None = None,
    schedule_sampling_probability: float = 1.0,
    generator: torch.Generator | None = None,
    rollout_steps: int | None = None,
) -> SequenceRollout:
    if perception_cfg.num_points != model.cfg.num_points:
        raise ValueError("Perception and CFM num_points must match")
    frame_count = batch["rgb"].shape[1]
    first = _frame_frontend(batch, 0, perception_cfg, generator)
    previous = FlowState(first.particles, first.velocity, first.confidence, torch.zeros_like(first.particles))
    particle_type = first.particle_type
    states: list[FlowState] = []
    losses: list[CFMLossOutput] = []
    valid_ratios = []
    loss_frames: set[int] = set()
    if loss_cfg is not None:
        loss_count = model.cfg.loss_frames_per_sequence
        if loss_count <= 0 or loss_count >= frame_count:
            loss_frames = set(range(frame_count))
        else:
            selected = torch.randperm(frame_count, generator=generator, device="cpu")[:loss_count]
            loss_frames = set(selected.tolist())

    for frame in range(frame_count):
        frontend = first if frame == 0 else _frame_frontend(batch, frame, perception_cfg, generator)
        image = prepare_image_condition(
            batch["rgb"][:, frame],
            batch["depth"][:, frame],
            model.cfg.image_size,
            perception_cfg.max_depth_m,
        )
        semantic = None
        if model.cfg.semantic_classes > 0:
            semantic = batch["semantic_image"][:, frame, 0]
        condition = CFMCondition(
            image=image,
            pose=batch["pose"][:, frame],
            observed_points=frontend.observed_points,
            observed_mask=frontend.observed_mask,
            previous_points=previous.position,
            previous_velocity=previous.velocity,
            confidence=previous.confidence,
            particle_type=particle_type,
            semantic=semantic,
        )
        if loss_cfg is not None and frame in loss_frames:
            losses.append(compute_cfm_loss(model, condition, batch["gt_points"], loss_cfg, generator))
        # The rollout state is detached before it becomes the next-frame
        # condition.  Building an autograd graph here only duplicates the ODE
        # graph already created by compute_cfm_loss and causes large sequence
        # memory growth.
        if loss_cfg is not None:
            with torch.no_grad():
                state = model(condition, num_steps=model.cfg.training_rollout_steps)
        else:
            state = model(condition, num_steps=rollout_steps)
        states.append(state)
        valid_ratios.append(frontend.valid_depth_ratio)

        # Scheduled sampling never substitutes GT. The alternative is a traditional
        # measurement-anchored prediction, which is easier early in training.
        use_raw_prediction = torch.rand((), generator=generator, device="cpu").item() < schedule_sampling_probability
        previous = (
            FlowState(
                state.position.detach(), state.velocity.detach(), state.confidence.detach(), state.acceleration.detach()
            )
            if use_raw_prediction
            else _anchor_prediction(state, particle_type, frontend.observed_points, frontend.observed_mask)
        )
    return SequenceRollout(states, losses, torch.stack(valid_ratios, dim=1), particle_type)
