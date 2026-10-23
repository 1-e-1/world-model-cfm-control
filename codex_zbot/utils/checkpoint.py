"""Checkpoint IO with architecture metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_class": model.__class__.__name__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "config": config,
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def load_checkpoint_resized_temporal_position(
    path: str | Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint while interpolating only temporal-position length.

    This is a narrow architecture migration for increasing an RGB-D window.
    All other parameter names and shapes remain strict, so it cannot silently
    hide unrelated model incompatibilities.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = payload["model"]
    target = model.state_dict()
    resized: list[str] = []
    for name, value in list(state.items()):
        if not name.endswith("temporal_position") or name not in target:
            continue
        expected = target[name]
        if value.shape == expected.shape:
            continue
        if value.ndim != 3 or expected.ndim != 3 or value.shape[0] != expected.shape[0] or value.shape[2] != expected.shape[2]:
            raise RuntimeError(
                f"cannot resize temporal position {name}: {tuple(value.shape)} -> {tuple(expected.shape)}"
            )
        state[name] = F.interpolate(
            value.transpose(1, 2), size=expected.shape[1], mode="linear", align_corners=True
        ).transpose(1, 2).to(dtype=expected.dtype)
        resized.append(name)
    model.load_state_dict(state, strict=True)
    payload["resized_temporal_position"] = resized
    return payload
