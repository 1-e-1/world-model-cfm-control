from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class OccupancyDriftConfig:
    num_points: int = 2048
    observed_points: int = 8192
    sequence_length: int = 16
    frame_stride: int = 2
    image_size: int = 128
    workspace_min: list[float] = field(default_factory=lambda: [-1.0, -1.0, 0.0])
    workspace_max: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    max_depth_m: float = 2.0
    observed_min_height_m: float = 0.01
    min_sequence_object_visibility: float = 0.005
    min_sequence_per_object_visibility: float = 0.0
    empty_frame_visibility_threshold: float = 1.0e-4
    max_sequence_empty_frame_fraction: float = 0.50
    # Training-label padding capacity only.  It is never passed to the model
    # or used as an inference-time object-count prior.
    max_objects: int = 5

    dim: int = 256
    heads: int = 4
    blocks: int = 3
    ffn_dim: int = 768
    context_tokens: int = 32
    image_tokens: int = 8
    pose_tokens: int = 4
    max_displacement_m: float = 0.24
    global_particle_ratio: float = 0.40
    global_prior_center: list[float] = field(default_factory=lambda: [0.5, 0.0, 0.1])

    generated_samples: int = 4
    positive_samples: int = 4
    voxel_resolutions: list[int] = field(default_factory=lambda: [8, 16])
    temperatures: list[float] = field(default_factory=lambda: [0.02, 0.05, 0.2])
    feature_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.25])
    drift_order: int = 1
    first_order_steps: int = 12
    first_order_step_size: float = 1.0 / 12
    second_order_eta1: float = 0.25
    second_order_eta2: float = 0.25
    second_order_steps: int = 4
    second_order_alpha: float = 3.0

    drift_weight: float = 0.25
    volume_chamfer_weight: float = 4.0
    outside_weight: float = 8.0
    observation_weight: float = 0.15
    repulsion_weight: float = 0.05
    match_points: int = 256
    occupancy_threshold_m: float = 0.05
    interior_margin_m: float = 0.01

    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    epochs: int = 100
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-2
    grad_clip: float = 2.0
    ema_decay: float = 0.995
    windows_per_episode_per_epoch: int = 1
    val_batches: int = 12
    early_stopping_patience: int = 20
    seed: int = 42

    # V7 uses this spatiotemporal architecture; its production dimensions are
    # explicit in occupancy_drifting_v7_partial_visibility.json.
    architecture: str = "v3_spatiotemporal"
    ray_grid_size: int = 8
    free_samples_per_ray: int = 3
    free_ray_end_fraction: float = 0.85
    memory_resolution: int = 12
    spatial_transformer_layers: int = 1
    temporal_transformer_layers: int = 2
    memory_cross_attention_layers: int = 2
    set_latent_tokens: int = 16
    coarse_temperature: float = 0.7
    coarse_cell_expansion: float = 0.75
    tri_state_weight: float = 1.0
    tri_state_class_weights: list[float] = field(default_factory=lambda: [5.0, 10.0, 0.1])
    tri_state_focal_gamma: float = 2.0
    free_space_weight: float = 1.5
    free_space_radius_m: float = 0.035
    implicit_occupancy_weight: float = 1.0
    implicit_sdf_weight: float = 0.5
    implicit_query_points: int = 384
    sdf_truncation_m: float = 0.15
    field_selection_temperature: float = 0.05
    coarse_warmup_epochs: int = 5
    coarse_teacher_forcing_epochs: int = 12

    def to_dict(self) -> dict:
        return asdict(self)


def load_drift_config(path: str | Path) -> OccupancyDriftConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return OccupancyDriftConfig(**payload)


def apply_v3_ablation(cfg: OccupancyDriftConfig, name: str) -> OccupancyDriftConfig:
    """Apply one named, checkpoint-reproducible V3 ablation in place."""
    if name == "full":
        return cfg
    if name == "no_temporal":
        cfg.temporal_transformer_layers = 0
    elif name == "no_free_space":
        cfg.free_space_weight = 0.0
    elif name == "no_implicit":
        cfg.implicit_occupancy_weight = 0.0
        cfg.implicit_sdf_weight = 0.0
    elif name == "local_only":
        cfg.global_particle_ratio = 0.0
    else:
        raise ValueError(f"Unknown V3 ablation: {name}")
    return cfg
