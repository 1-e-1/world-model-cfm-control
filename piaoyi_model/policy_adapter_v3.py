from __future__ import annotations

import torch

from .inference_v3 import OccupancyDriftV3State


def v3_policy_features(
    current: OccupancyDriftV3State,
    previous: OccupancyDriftV3State | None = None,
) -> torch.Tensor:
    """Nine policy-facing active-perception statistics.

    The quantities are reconstruction confidence signals, not physical point
    velocity.  Deltas are zero on the first observation window.
    """
    observed_fraction = current.observed_mask.float().mean(dim=1, keepdim=True)
    spread = current.position.std(dim=1).mean(dim=1, keepdim=True)
    mean_height = current.position[..., 2].mean(dim=1, keepdim=True)
    if previous is None:
        uncertainty_gain = torch.zeros_like(current.ensemble_uncertainty)
        unknown_reduction = torch.zeros_like(current.unknown_fraction)
        centroid_shift = torch.zeros_like(current.ensemble_uncertainty)
    else:
        uncertainty_gain = previous.ensemble_uncertainty - current.ensemble_uncertainty
        unknown_reduction = previous.unknown_fraction - current.unknown_fraction
        centroid_shift = (current.position.mean(dim=1) - previous.position.mean(dim=1)).norm(dim=1, keepdim=True)
    return torch.cat([
        current.ensemble_uncertainty,
        current.unknown_fraction,
        current.free_space_conflict,
        observed_fraction,
        spread,
        mean_height,
        uncertainty_gain,
        unknown_reduction,
        centroid_shift,
    ], dim=1)
