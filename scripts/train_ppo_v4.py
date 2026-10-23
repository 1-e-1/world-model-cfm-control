"""Train recurrent PPO with frozen V4 drift and belief World Model."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsl_rl.runners import OnPolicyRunner
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config", default=str(ROOT / "configs/formal_v4.json"))
parser.add_argument("--drift_config", default=str(ROOT / "piaoyi_model/configs/occupancy_drifting_v7_partial_visibility.json"))
parser.add_argument("--drift_checkpoint", required=True)
parser.add_argument("--world_checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--resume", default=None)
parser.add_argument("--run_name", default="codex_zbot_active_vision_v4")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from codex_zbot.config import load_config
from codex_zbot.policy.rsl_rl_cfg import ActiveVisionPPORunnerCfg
from codex_zbot.tasks.active_vision_v4_env import ActiveVisionV4Env, ActiveVisionV4EnvCfg


def main() -> None:
    cfg = load_config(args_cli.config)
    if args_cli.num_envs is not None:
        cfg.ppo.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        cfg.ppo.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        cfg.ppo.seed = args_cli.seed
    env_cfg = ActiveVisionV4EnvCfg()
    env_cfg.scene.num_envs = cfg.ppo.num_envs
    env_cfg.seed = cfg.ppo.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.algorithm_config_path = str(Path(args_cli.config).resolve())
    env_cfg.drift_config_path = str(Path(args_cli.drift_config).resolve())
    env_cfg.drift_checkpoint = str(Path(args_cli.drift_checkpoint).resolve())
    env_cfg.world_checkpoint = str(Path(args_cli.world_checkpoint).resolve())
    env_cfg.decimation = cfg.ppo.decimation
    env_cfg.episode_length_s = cfg.ppo.episode_length_s
    env_cfg.sim.dt = cfg.ppo.physics_dt
    env_cfg.sim.render_interval = env_cfg.decimation

    agent_cfg = ActiveVisionPPORunnerCfg()
    agent_cfg.experiment_name = args_cli.run_name
    agent_cfg.num_steps_per_env = cfg.ppo.rollout_steps
    agent_cfg.max_iterations = cfg.ppo.max_iterations
    agent_cfg.seed = cfg.ppo.seed
    agent_cfg.algorithm.learning_rate = cfg.ppo.learning_rate
    agent_cfg.algorithm.gamma = cfg.ppo.gamma
    agent_cfg.algorithm.lam = cfg.ppo.lam
    agent_cfg.algorithm.clip_param = cfg.ppo.clip_param
    agent_cfg.algorithm.entropy_coef = cfg.ppo.entropy_coef
    agent_cfg.algorithm.num_learning_epochs = cfg.ppo.learning_epochs
    agent_cfg.algorithm.num_mini_batches = min(cfg.ppo.mini_batches, cfg.ppo.num_envs)
    agent_cfg.actor.rnn_hidden_dim = cfg.ppo.recurrent_hidden_dim
    agent_cfg.critic.rnn_hidden_dim = cfg.ppo.recurrent_hidden_dim
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    run_dir = ROOT / cfg.log_dir / "rsl_rl" / args_cli.run_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    launch = {
        "algorithm_config": str(Path(args_cli.config).resolve()),
        "drift_config": env_cfg.drift_config_path,
        "drift_checkpoint": env_cfg.drift_checkpoint,
        "world_checkpoint": env_cfg.world_checkpoint,
        "num_envs": cfg.ppo.num_envs,
        "control_hz": 1.0 / (cfg.ppo.physics_dt * cfg.ppo.decimation),
        "reconstruction_hz": 1.0 / (
            cfg.ppo.physics_dt * cfg.ppo.decimation * cfg.ppo.reconstruction_interval
        ),
        "seed": cfg.ppo.seed,
        "device": args_cli.device,
    }
    (run_dir / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    (run_dir / "algorithm_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    env = ActiveVisionV4Env(env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)
    if args_cli.resume:
        runner.load(str(Path(args_cli.resume).resolve()))
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    non_finite = [
        f"{module_name}.{name}"
        for module_name, module in (("actor", runner.alg.actor), ("critic", runner.alg.critic))
        for name, parameter in module.named_parameters()
        if not parameter.isfinite().all()
    ]
    if non_finite:
        raise FloatingPointError(f"V4 PPO produced non-finite parameters: {non_finite[:8]}")
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda path: path.stat().st_mtime)
    completion = {"complete": True, "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None}
    (run_dir / "training_complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
