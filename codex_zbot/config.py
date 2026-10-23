"""Typed, JSON-serializable experiment configuration."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar


@dataclass
class PerceptionConfig:
    num_points: int = 2048
    observed_points: int = 1024
    max_depth_m: float = 2.0
    voxel_size_m: float = 0.01
    outlier_radius_m: float = 0.04
    outlier_min_neighbors: int = 2
    workspace_min: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    workspace_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    min_valid_depth_ratio: float = 0.15
    latent_particle_ratio: float = 0.35


@dataclass
class CFMConfig:
    dynamics_order: int = 2
    num_points: int = 2048
    observed_points: int = 1024
    image_size: int = 64
    image_channels: int = 5  # RGB + depth + valid-depth mask
    semantic_classes: int = 0
    dim: int = 384
    heads: int = 6
    blocks: int = 6
    ffn_dim: int = 1024
    image_tokens: int = 16
    point_tokens: int = 16
    pose_tokens: int = 4
    state_tokens: int = 4
    ode_steps: int = 8
    training_rollout_steps: int = 2
    vmax: float = 1.0
    solver: str = "midpoint"
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    epochs: int = 100
    sequence_length: int = 4
    loss_frames_per_sequence: int = 1
    windows_per_episode_per_epoch: int = 1
    schedule_sampling_max: float = 0.7
    amp_dtype: str = "bfloat16"
    grad_clip: float = 1.0
    seed: int = 42


@dataclass
class CFMLossConfig:
    acceleration: float = 1.0
    terminal_position: float = 2.0
    terminal_velocity: float = 0.1
    terminal_acceleration: float = 0.05
    observation_consistency: float = 2.0
    repulsion: float = 0.05
    confidence: float = 0.1
    terminal_steps: int = 2
    match_points: int = 512
    sinkhorn_epsilon: float = 0.03
    sinkhorn_iterations: int = 20


@dataclass
class WorldModelConfig:
    enabled: bool = True
    latent_dim: int = 128
    robot_state_dim: int = 23
    action_dim: int = 8
    hidden_dim: int = 256
    recurrent_dim: int = 128
    sequence_length: int = 8
    rollout_horizon: int = 5
    lr: float = 3.0e-4
    # CFM rollouts are part of each world-model batch; keep this conservative.
    batch_size: int = 4
    windows_per_episode_per_epoch: int = 1
    epochs: int = 100
    grad_clip: float = 1.0
    seed: int = 42


@dataclass
class PPOConfig:
    num_envs: int = 16
    physics_dt: float = 1.0 / 60.0
    decimation: int = 4
    episode_length_s: float = 15.0
    action_scale_rad: float = 0.05
    action_filter_alpha: float = 0.2
    joint_limit_margin_rad: float = 0.174533
    safe_joint_excursion_rad: float = 0.35
    camera_object_hard_distance_m: float = 0.40
    camera_object_soft_distance_m: float = 0.55
    camera_workcell_x_max_m: float = 0.40
    camera_z_hard_min_m: float = 0.25
    camera_z_soft_min_m: float = 0.36
    camera_safety_hysteresis_m: float = 0.10
    kinematic_control: bool = True
    min_object_pixel_ratio: float = 0.005
    informative_object_pixel_ratio: float = 0.015
    min_per_object_pixel_ratio: float = 0.001
    informative_per_object_pixel_ratio: float = 0.003
    max_missing_object_frames: int = 3
    rollout_steps: int = 32
    max_iterations: int = 500
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    learning_epochs: int = 5
    mini_batches: int = 4
    recurrent_hidden_dim: int = 128
    reconstruction_interval: int = 2
    drift_ensemble_samples: int = 2
    seed: int = 42


@dataclass
class RewardConfig:
    coverage_gain: float = 1.0
    reconstruction_gain: float = 1.0
    uncertainty_gain: float = 0.3
    novel_view: float = 0.2
    visibility: float = 0.1
    flow_convergence: float = 0.1
    unknown_reduction: float = 0.3
    action_l2: float = -0.01
    action_rate: float = -0.05
    joint_velocity: float = -0.01
    joint_margin: float = -0.2
    collision: float = -2.0
    unsafe_termination: float = -5.0
    camera_proximity: float = -2.0
    success_bonus: float = 5.0
    coverage_reference_gain: float = 0.01
    chamfer_reference_gain_m: float = 0.01
    uncertainty_reference_gain: float = 0.005
    unknown_reference_gain: float = 0.01
    centroid_shift_reference_m: float = 0.01
    visibility_reference: float = 0.008
    success_coverage: float = 0.90
    convergence_speed_m: float = 0.01
    convergence_steps: int = 5


@dataclass
class ExperimentConfig:
    name: str = "zbot_sequential_active_vision"
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    cfm: CFMConfig = field(default_factory=CFMConfig)
    cfm_loss: CFMLossConfig = field(default_factory=CFMLossConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


T = TypeVar("T")


def _merge_dataclass(instance: T, values: dict[str, Any]) -> T:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path is None:
        return cfg
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return _merge_dataclass(cfg, values)
