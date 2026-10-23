"""Stable alternating training orchestrator.

Each round keeps CFM and the World Model frozen throughout a PPO block, then collects
on-policy replay, performs supervised CFM updates, and finally refreshes belief dynamics.
No reward model changes inside an on-policy rollout.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("[JOINT]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def newest_ppo_checkpoint() -> Path:
    candidates = list((ROOT / "logs" / "rsl_rl" / "codex_zbot_active_vision").glob("**/model_*.pt"))
    if not candidates:
        raise RuntimeError("PPO block completed without producing a model checkpoint")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_data_dir", required=True)
    parser.add_argument("--cfm_checkpoint", required=True)
    parser.add_argument("--belief_checkpoint", required=True)
    parser.add_argument("--ppo_checkpoint", default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--ppo_iterations", type=int, default=50)
    parser.add_argument("--cfm_epochs", type=int, default=2)
    parser.add_argument("--world_epochs", type=int, default=2)
    parser.add_argument("--replay_episodes", type=int, default=32)
    parser.add_argument("--num_envs", type=int, default=16)
    args = parser.parse_args()

    python = sys.executable
    cfm = Path(args.cfm_checkpoint).resolve()
    belief = Path(args.belief_checkpoint).resolve()
    ppo = Path(args.ppo_checkpoint).resolve() if args.ppo_checkpoint else None
    manifest_dir = ROOT / "logs" / "joint" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for round_id in range(args.rounds):
        ppo_command = [
            python,
            "-u",
            str(ROOT / "scripts" / "train_ppo.py"),
            "--headless",
            "--enable_cameras",
            "--config",
            args.config,
            "--cfm_checkpoint",
            str(cfm),
            "--belief_checkpoint",
            str(belief),
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.ppo_iterations),
        ]
        if ppo is not None:
            ppo_command.extend(["--resume", str(ppo)])
        run(ppo_command)
        ppo = newest_ppo_checkpoint()

        replay_dir = ROOT / "data" / "joint_replay" / f"round_{round_id:03d}"
        run(
            [
                python,
                "-u",
                str(ROOT / "scripts" / "collect_on_policy.py"),
                "--headless",
                "--enable_cameras",
                "--config",
                args.config,
                "--cfm_checkpoint",
                str(cfm),
                "--belief_checkpoint",
                str(belief),
                "--ppo_checkpoint",
                str(ppo),
                "--out_dir",
                str(replay_dir),
                "--episodes",
                str(args.replay_episodes),
                "--num_envs",
                str(min(args.num_envs, 8)),
            ]
        )

        run(
            [
                python,
                "-u",
                str(ROOT / "scripts" / "train_cfm.py"),
                "--config",
                args.config,
                "--data_dir",
                args.base_data_dir,
                "--extra_data_dir",
                str(replay_dir),
                "--run_name",
                "joint_cfm",
                "--max_epochs",
                str(args.cfm_epochs),
                "--resume",
                str(cfm),
            ]
        )
        cfm = ROOT / "checkpoints" / "joint_cfm" / "last.pt"
        run(
            [
                python,
                "-u",
                str(ROOT / "scripts" / "train_world_model.py"),
                "--config",
                args.config,
                "--cfm_checkpoint",
                str(cfm),
                "--data_dir",
                args.base_data_dir,
                "--extra_data_dir",
                str(replay_dir),
                "--run_name",
                "joint_world_model",
                "--max_epochs",
                str(args.world_epochs),
                "--resume",
                str(belief),
            ]
        )
        belief = ROOT / "checkpoints" / "joint_world_model" / "last.pt"
        history.append(
            {
                "round": round_id,
                "ppo": str(ppo),
                "cfm": str(cfm),
                "belief": str(belief),
                "replay": str(replay_dir),
            }
        )
        (manifest_dir / "manifest.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[JOINT] complete; manifest={manifest_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

