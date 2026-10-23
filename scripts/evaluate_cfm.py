"""Evaluate sequential CFM on held-out scene seeds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from codex_zbot.cfm import SecondOrderCFM
from codex_zbot.cfm.rollout import rollout_sequence
from codex_zbot.config import load_config
from codex_zbot.data import SequentialEpisodeDataset
from codex_zbot.utils.checkpoint import load_checkpoint
from codex_zbot.utils.metrics import chamfer_distance, coverage_at_threshold
from codex_zbot.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_batches", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.perception.num_points = cfg.cfm.num_points
    cfg.perception.observed_points = cfg.cfm.observed_points
    generator = seed_everything(cfg.cfm.seed)
    device = torch.device(args.device)
    dataset = SequentialEpisodeDataset(
        args.data_dir,
        args.split,
        cfg.cfm.sequence_length,
        num_gt_points=cfg.cfm.num_points,
        image_size=cfg.cfm.image_size,
        seed=cfg.cfm.seed,
        min_first_valid_ratio=cfg.perception.min_valid_depth_ratio,
    )
    loader = DataLoader(dataset, batch_size=cfg.cfm.batch_size, shuffle=False)
    model = SecondOrderCFM(cfg.cfm).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    metrics = {"chamfer": [], "coverage_005": [], "coverage_010": [], "flow_speed": []}
    with torch.inference_mode():
        for batch_id, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            rollout = rollout_sequence(model, batch, cfg.perception, None, 1.0, generator)
            state = rollout.states[-1]
            metrics["chamfer"].extend(chamfer_distance(state.position, batch["gt_points"]).cpu().tolist())
            metrics["coverage_005"].extend(
                coverage_at_threshold(state.position, batch["gt_points"], 0.05).cpu().tolist()
            )
            metrics["coverage_010"].extend(
                coverage_at_threshold(state.position, batch["gt_points"], 0.10).cpu().tolist()
            )
            metrics["flow_speed"].extend(state.velocity.norm(dim=-1).mean(dim=1).cpu().tolist())
            if batch_id + 1 >= args.max_batches:
                break
    summary = {
        key: {"mean": sum(values) / len(values), "count": len(values)} for key, values in metrics.items() if values
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
