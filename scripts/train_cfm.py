"""Train the sequential second-order CFM without starting Isaac Sim."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from codex_zbot.cfm import SecondOrderCFM
from codex_zbot.cfm.rollout import rollout_sequence
from codex_zbot.config import load_config
from codex_zbot.data import SequentialEpisodeDataset, split_episode_paths
from codex_zbot.utils.checkpoint import load_checkpoint, save_checkpoint
from codex_zbot.utils.seed import seed_everything, seed_worker


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, cfg, device, generator, max_batches: int = 8) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        rollout = rollout_sequence(model, batch, cfg.perception, cfg.cfm_loss, 1.0, generator)
        for loss in rollout.losses:
            for key, value in loss.detached_metrics().items():
                totals[key] = totals.get(key, 0.0) + value
        batches += len(rollout.losses)
        if batches >= max_batches:
            break
    return {key: value / max(1, batches) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.json"))
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--extra_data_dir", action="append", default=[])
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.data_dir = str(Path(args.data_dir).resolve())
    if args.max_epochs is not None:
        cfg.cfm.epochs = args.max_epochs
    cfg.perception.num_points = cfg.cfm.num_points
    cfg.perception.observed_points = cfg.cfm.observed_points
    generator = seed_everything(cfg.cfm.seed)
    device = torch.device(args.device)

    train_paths = split_episode_paths(cfg.data_dir, "train", cfg.cfm.seed)
    for extra_dir in args.extra_data_dir:
        train_paths.extend(split_episode_paths(extra_dir, "train", cfg.cfm.seed))
    train_data = SequentialEpisodeDataset(
        cfg.data_dir,
        "train",
        cfg.cfm.sequence_length,
        num_gt_points=cfg.cfm.num_points,
        image_size=cfg.cfm.image_size,
        seed=cfg.cfm.seed,
        min_first_valid_ratio=cfg.perception.min_valid_depth_ratio,
        max_windows_per_episode=cfg.cfm.windows_per_episode_per_epoch,
        paths=train_paths,
    )
    val_data = SequentialEpisodeDataset(
        cfg.data_dir,
        "val",
        cfg.cfm.sequence_length,
        num_gt_points=cfg.cfm.num_points,
        image_size=cfg.cfm.image_size,
        seed=cfg.cfm.seed,
        min_first_valid_ratio=cfg.perception.min_valid_depth_ratio,
    )
    loader_kwargs = dict(
        batch_size=cfg.cfm.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_kwargs)

    model = SecondOrderCFM(cfg.cfm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.cfm.lr, weight_decay=cfg.cfm.weight_decay)
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", -1)) + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.cfm.epochs, eta_min=cfg.cfm.lr * 0.1)
    run_name = args.run_name or cfg.name + "_cfm"
    log_dir = ROOT / cfg.log_dir / run_name
    checkpoint_dir = ROOT / cfg.checkpoint_dir / run_name
    writer = SummaryWriter(log_dir)
    (log_dir / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    global_step = 0
    best_val = float("inf")
    best_path = checkpoint_dir / "best.pt"
    if args.resume and best_path.exists():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        best_val = float(best_payload.get("metrics", {}).get("total", best_val))
    end_epoch = start_epoch + cfg.cfm.epochs
    for epoch in range(start_epoch, end_epoch):
        train_data.set_epoch(epoch)
        model.train()
        local_epoch = epoch - start_epoch
        schedule_probability = cfg.cfm.schedule_sampling_max * local_epoch / max(1, cfg.cfm.epochs - 1)
        epoch_total = 0.0
        epoch_steps = 0
        accumulation = max(1, cfg.cfm.gradient_accumulation_steps)
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            batch = move_batch(batch, device)
            amp_enabled = device.type == "cuda"
            amp_dtype = torch.bfloat16 if cfg.cfm.amp_dtype == "bfloat16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                rollout = rollout_sequence(
                    model, batch, cfg.perception, cfg.cfm_loss, schedule_probability, generator
                )
                loss = torch.stack([item.total for item in rollout.losses]).mean()
            (loss / accumulation).backward()
            epoch_total += float(loss.detach())
            epoch_steps += 1
            writer.add_scalar("train/loss", loss.detach(), global_step)
            if epoch_steps % accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.cfm.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
            writer.add_scalar("train/schedule_sampling", schedule_probability, global_step)
            global_step += 1
            if args.max_batches is not None and epoch_steps >= args.max_batches:
                break
        if epoch_steps == 0:
            raise RuntimeError(
                "CFM train loader produced no batches; collect more valid sequences or reduce cfm.batch_size"
            )
        if epoch_steps % accumulation != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.cfm.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)
        scheduler.step()

        metrics = evaluate(model, val_loader, cfg, device, generator)
        train_average = epoch_total / max(1, epoch_steps)
        writer.add_scalar("epoch/train_loss", train_average, epoch)
        writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], epoch)
        for key, value in metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        print(
            f"[CFM] epoch={epoch:03d} train={train_average:.5f} "
            f"val={metrics.get('total', float('nan')):.5f} p_model={schedule_probability:.3f}"
        )
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, cfg.to_dict(), metrics)
        validation = metrics.get("total", float("inf"))
        if validation < best_val:
            best_val = validation
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, cfg.to_dict(), metrics)
    writer.close()


if __name__ == "__main__":
    main()
