from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .conditioning import collect_observed_object_points, image_condition
from .config import OccupancyDriftConfig
from .evidence import build_ray_evidence
from .features import soft_voxel_occupancy
from .model_v3 import OccupancyDriftGeneratorV3, V3Memory


@dataclass
class OccupancyDriftV3State:
    position: torch.Tensor
    ensemble_uncertainty: torch.Tensor
    unknown_fraction: torch.Tensor
    free_space_conflict: torch.Tensor
    observed_points: torch.Tensor
    observed_mask: torch.Tensor
    tri_state_probability: torch.Tensor
    nfe: int = 1
    memory: V3Memory | None = None


@torch.no_grad()
def reconstruct_occupancy_v3(
    model: OccupancyDriftGeneratorV3,
    batch: dict[str, torch.Tensor],
    cfg: OccupancyDriftConfig,
    generator: torch.Generator | None = None,
    ensemble_samples: int = 4,
    selection_temperature: float | None = None,
) -> OccupancyDriftV3State:
    observed, observed_mask = collect_observed_object_points(batch, cfg, generator)
    evidence = build_ray_evidence(batch, cfg)
    noise = torch.randn(
        batch["rgb"].shape[0], ensemble_samples, cfg.num_points, 3, generator=generator, device="cpu"
    ).to(batch["rgb"].device)
    output = model(
        noise, image_condition(batch, cfg.max_depth_m), batch["pose"], evidence,
        observed, observed_mask, return_aux=True,
    )
    lower = torch.tensor(cfg.workspace_min, device=noise.device).view(1, 1, 1, 3)
    upper = torch.tensor(cfg.workspace_max, device=noise.device).view(1, 1, 1, 3)
    occupancy = soft_voxel_occupancy(output.points, cfg.voxel_resolutions[0], lower, upper)
    uncertainty = occupancy.var(dim=1, unbiased=False).mean(dim=1, keepdim=True)
    # The implicit field ranks all ensemble candidates. This uses the same
    # encoded memory and is still a single generator evaluation (NFE=1).
    candidates = output.points.flatten(1, 2)
    candidate_score, _ = model.decode_field(output.memory, candidates)
    temperature = cfg.field_selection_temperature if selection_temperature is None else selection_temperature
    if temperature <= 0:
        priority = candidate_score
    else:
        # Use an independent stream: reusing a noise coordinate would bias
        # selection toward specific coarse CDF intervals.
        selection_uniform = torch.rand(
            candidates.shape[:2], generator=generator, device="cpu", dtype=candidates.dtype
        ).to(candidates.device).clamp(1.0e-6, 1.0 - 1.0e-6)
        gumbel = -torch.log(-torch.log(selection_uniform))
        priority = F.logsigmoid(candidate_score) / temperature + gumbel
    selected_ids = priority.topk(k=min(cfg.num_points, candidates.shape[1]), dim=1).indices
    selected = torch.gather(candidates, 1, selected_ids[..., None].expand(-1, -1, 3))
    tri_probability = output.memory.tri_state_logits.softmax(dim=-1)
    unknown_fraction = tri_probability[..., 2].mean(dim=1, keepdim=True)
    generated = selected
    free = evidence.free_points.flatten(1, 2)
    free_mask = evidence.free_mask.flatten(1, 2)
    distance = torch.cdist(generated, free).masked_fill(~free_mask[:, None], float("inf")).amin(dim=2)
    free_conflict = (distance < cfg.free_space_radius_m).float().mean(dim=1, keepdim=True)
    return OccupancyDriftV3State(
        selected, uncertainty, unknown_fraction, free_conflict,
        observed, observed_mask, tri_probability, memory=output.memory,
    )
