"""Train reconstruction belief encoding and action-conditioned dynamics from CFM rollouts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from codex_zbot.cfm import SecondOrderCFM
from codex_zbot.cfm.rollout import rollout_sequence
from codex_zbot.config import load_config
from codex_zbot.data import SequentialEpisodeDataset, split_episode_paths
from codex_zbot.policy.rewards import depth_consistent_visibility
from codex_zbot.utils.checkpoint import load_checkpoint, save_checkpoint
from codex_zbot.utils.metrics import chamfer_distance
from codex_zbot.utils.seed import seed_everything, seed_worker
from codex_zbot.world_model import BeliefLearningSystem
from codex_zbot.world_model.model import world_model_loss


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def belief_targets(batch, rollout, system, control_dt: float, action_scale: float):
    positions = [state.position.detach() for state in rollout.states]
    latents = torch.stack(
        [
            system.encoder(
                state.position.detach(),
                state.velocity.detach(),
                state.confidence.detach(),
                rollout.particle_type,
                positions[max(0, index - 1)],
                state.acceleration.detach(),
            )
            for index, state in enumerate(rollout.states)
        ],
        dim=1,
    )
    chamfer = torch.stack([chamfer_distance(state.position.detach(), batch["gt_points"]) for state in rollout.states], dim=1)
    flow_speed = torch.stack([state.velocity.detach().norm(dim=-1).mean(dim=1) for state in rollout.states], dim=1)

    seen = torch.zeros(
        batch["gt_points"].shape[:2], device=batch["gt_points"].device, dtype=torch.bool
    )
    coverage_values = []
    for frame in range(batch["rgb"].shape[1]):
        visible = depth_consistent_visibility(
            batch["gt_points"],
            batch["depth"][:, frame].permute(0, 2, 3, 1),
            batch["intrinsics"],
            batch["pose"][:, frame],
        )
        seen |= visible
        coverage_values.append(seen.float().mean(dim=1))
    coverage = torch.stack(coverage_values, dim=1)

    joint = batch["joint_pos"]
    joint_velocity = batch.get("joint_vel", torch.zeros_like(joint))
    if torch.count_nonzero(joint_velocity) == 0:
        joint_velocity[:, 1:] = (joint[:, 1:] - joint[:, :-1]) / control_dt
    robot_state = torch.cat([joint, joint_velocity, batch["pose"]], dim=2)
    actions = ((joint[:, 1:] - joint[:, :-1]) / action_scale).clamp(-1.0, 1.0)
    metric_targets = torch.stack([chamfer, coverage, flow_speed], dim=-1)
    return {
        "latents": latents,
        "robot_state": robot_state[:, :-1],
        "actions": actions,
        "coverage_gain": coverage[:, 1:] - coverage[:, :-1],
        "reconstruction_gain": chamfer[:, :-1] - chamfer[:, 1:],
        "collision": batch.get("collision", torch.zeros_like(coverage, dtype=torch.bool))[:, 1:],
        "metric_targets": metric_targets,
    }


def compute_batch_loss(batch, cfm, system, cfg, generator):
    with torch.no_grad():
        rollout = rollout_sequence(
            cfm,
            batch,
            cfg.perception,
            None,
            1.0,
            generator,
            rollout_steps=cfg.cfm.training_rollout_steps,
        )
    control_dt = cfg.ppo.physics_dt * cfg.ppo.decimation
    targets = belief_targets(batch, rollout, system, control_dt, cfg.ppo.action_scale_rad)
    predictions = system.dynamics.rollout(
        targets["latents"][:, 0], targets["robot_state"], targets["actions"]
    )
    dynamics_loss, dynamics_terms = world_model_loss(
        predictions,
        targets["latents"][:, 1:].detach(),
        targets["coverage_gain"],
        targets["reconstruction_gain"],
        targets["collision"],
    )
    decoded_metrics = system.metric_head(targets["latents"])
    metric_loss = F.huber_loss(decoded_metrics, targets["metric_targets"])
    latent_std = targets["latents"].flatten(0, 1).std(dim=0)
    variance_loss = F.relu(0.1 - latent_std).mean()
    loss = dynamics_loss + metric_loss + 0.1 * variance_loss
    return loss, dynamics_terms, metric_loss, variance_loss


@torch.no_grad()
def evaluate(system, cfm, loader, cfg, device, generator, max_batches: int = 4) -> float:
    system.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        loss, _, _, _ = compute_batch_loss(batch, cfm, system, cfg, generator)
        total += float(loss)
        count += 1
        if count >= max_batches:
            break
    if count == 0:
        raise RuntimeError("World-model validation loader produced no batches")
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cfm_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--extra_data_dir", action="append", default=[])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--val_batches", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.world_model.enabled:
        raise ValueError("world_model.enabled=false; there is no World Model to train")
    cfg.perception.num_points = cfg.cfm.num_points
    cfg.perception.observed_points = cfg.cfm.observed_points
    if args.max_epochs is not None:
        cfg.world_model.epochs = args.max_epochs
    generator = seed_everything(cfg.world_model.seed)
    device = torch.device(args.device)
    sequence_length = cfg.world_model.sequence_length + 1
    train_paths = split_episode_paths(args.data_dir, "train", cfg.world_model.seed)
    for extra_dir in args.extra_data_dir:
        train_paths.extend(split_episode_paths(extra_dir, "train", cfg.world_model.seed))
    dataset = SequentialEpisodeDataset(
        args.data_dir,
        "train",
        sequence_length,
        num_gt_points=cfg.cfm.num_points,
        image_size=cfg.cfm.image_size,
        seed=cfg.world_model.seed,
        min_first_valid_ratio=cfg.perception.min_valid_depth_ratio,
        max_windows_per_episode=cfg.world_model.windows_per_episode_per_epoch,
        paths=train_paths,
    )
    val_dataset = SequentialEpisodeDataset(
        args.data_dir,
        "val",
        sequence_length,
        num_gt_points=cfg.cfm.num_points,
        image_size=cfg.cfm.image_size,
        seed=cfg.world_model.seed,
        min_first_valid_ratio=cfg.perception.min_valid_depth_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.world_model.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.world_model.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    cfm = SecondOrderCFM(cfg.cfm).to(device)
    load_checkpoint(args.cfm_checkpoint, cfm, map_location=device)
    cfm.requires_grad_(False).eval()
    system = BeliefLearningSystem(cfg.world_model).to(device)
    optimizer = torch.optim.AdamW(system.parameters(), lr=cfg.world_model.lr, weight_decay=1.0e-5)
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(args.resume, system, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", -1)) + 1
    run_name = args.run_name or cfg.name + "_world_model"
    log_dir = ROOT / cfg.log_dir / run_name
    checkpoint_dir = ROOT / cfg.checkpoint_dir / run_name
    writer = SummaryWriter(log_dir)
    (log_dir / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    global_step = 0
    best_val = float("inf")
    best_path = checkpoint_dir / "best.pt"
    if args.resume and best_path.exists():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        best_val = float(best_payload.get("metrics", {}).get("val_loss", best_val))
    for epoch in range(start_epoch, start_epoch + cfg.world_model.epochs):
        dataset.set_epoch(epoch)
        system.train()
        total_epoch = 0.0
        count = 0
        for batch in loader:
            batch = move_batch(batch, device)
            loss, dynamics_terms, metric_loss, variance_loss = compute_batch_loss(
                batch, cfm, system, cfg, generator
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(system.parameters(), cfg.world_model.grad_clip)
            optimizer.step()
            total_epoch += float(loss.detach())
            count += 1
            writer.add_scalar("train/total", loss.detach(), global_step)
            writer.add_scalar("train/metric", metric_loss.detach(), global_step)
            writer.add_scalar("train/variance", variance_loss.detach(), global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)
            for key, value in dynamics_terms.items():
                writer.add_scalar(f"train/{key}", value.detach(), global_step)
            global_step += 1
            if args.max_batches is not None and count >= args.max_batches:
                break
        if count == 0:
            raise RuntimeError(
                "World-model train loader produced no batches; collect more valid sequences or reduce "
                "world_model.batch_size"
            )
        average = total_epoch / max(1, count)
        val_loss = evaluate(system, cfm, val_loader, cfg, device, generator, args.val_batches)
        writer.add_scalar("epoch/train_loss", average, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        print(f"[WORLD] epoch={epoch:03d} train={average:.5f} val={val_loss:.5f}")
        metrics = {"train_loss": average, "val_loss": val_loss}
        save_checkpoint(checkpoint_dir / "last.pt", system, optimizer, epoch, cfg.to_dict(), metrics)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(checkpoint_dir / "best.pt", system, optimizer, epoch, cfg.to_dict(), metrics)
    writer.close()


if __name__ == "__main__":
    main()
