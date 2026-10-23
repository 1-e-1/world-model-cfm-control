"""Train recurrent PPO with frozen CFM and frozen belief world model."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Native extension must be imported before carb on Windows.
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config", default=str(ROOT / "configs" / "base.json"))
parser.add_argument("--cfm_checkpoint", required=True)
parser.add_argument("--belief_checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--resume", default=None)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from codex_zbot.config import load_config
from codex_zbot.policy.rsl_rl_cfg import ActiveVisionPPORunnerCfg
from codex_zbot.tasks.active_vision_env import ActiveVisionEnv, ActiveVisionEnvCfg


def main() -> None:
    algorithm_cfg = load_config(args_cli.config)
    if args_cli.num_envs is not None:
        algorithm_cfg.ppo.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        algorithm_cfg.ppo.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        algorithm_cfg.ppo.seed = args_cli.seed

    env_cfg = ActiveVisionEnvCfg()
    env_cfg.scene.num_envs = algorithm_cfg.ppo.num_envs
    env_cfg.seed = algorithm_cfg.ppo.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.algorithm_config_path = str(Path(args_cli.config).resolve())
    env_cfg.cfm_checkpoint = str(Path(args_cli.cfm_checkpoint).resolve())
    env_cfg.belief_checkpoint = str(Path(args_cli.belief_checkpoint).resolve())
    env_cfg.decimation = algorithm_cfg.ppo.decimation
    env_cfg.episode_length_s = algorithm_cfg.ppo.episode_length_s
    env_cfg.sim.dt = algorithm_cfg.ppo.physics_dt
    env_cfg.sim.render_interval = env_cfg.decimation

    agent_cfg = ActiveVisionPPORunnerCfg()
    agent_cfg.num_steps_per_env = algorithm_cfg.ppo.rollout_steps
    agent_cfg.max_iterations = algorithm_cfg.ppo.max_iterations
    agent_cfg.seed = algorithm_cfg.ppo.seed
    agent_cfg.algorithm.learning_rate = algorithm_cfg.ppo.learning_rate
    agent_cfg.algorithm.gamma = algorithm_cfg.ppo.gamma
    agent_cfg.algorithm.lam = algorithm_cfg.ppo.lam
    agent_cfg.algorithm.clip_param = algorithm_cfg.ppo.clip_param
    agent_cfg.algorithm.entropy_coef = algorithm_cfg.ppo.entropy_coef
    agent_cfg.algorithm.num_learning_epochs = algorithm_cfg.ppo.learning_epochs
    # RSL-RL recurrent batching partitions along the environment dimension.  A
    # larger mini-batch count silently creates empty batches and NaN losses.
    agent_cfg.algorithm.num_mini_batches = min(
        algorithm_cfg.ppo.mini_batches, algorithm_cfg.ppo.num_envs
    )
    agent_cfg.actor.rnn_hidden_dim = algorithm_cfg.ppo.recurrent_hidden_dim
    agent_cfg.critic.rnn_hidden_dim = algorithm_cfg.ppo.recurrent_hidden_dim
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    run_dir = ROOT / algorithm_cfg.log_dir / "rsl_rl" / agent_cfg.experiment_name / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "algorithm_config.json").write_text(
        json.dumps(algorithm_cfg.to_dict(), indent=2), encoding="utf-8"
    )
    (run_dir / "launch.json").write_text(
        json.dumps(
            {
                "cfm_checkpoint": env_cfg.cfm_checkpoint,
                "belief_checkpoint": env_cfg.belief_checkpoint,
                "num_envs": env_cfg.scene.num_envs,
                "device": args_cli.device,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    env = ActiveVisionEnv(env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)
    if args_cli.resume:
        runner.load(str(Path(args_cli.resume).resolve()))
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    non_finite = [
        name
        for module_name, module in (("actor", runner.alg.actor), ("critic", runner.alg.critic))
        for name, parameter in module.named_parameters()
        if not parameter.isfinite().all()
    ]
    if non_finite:
        raise FloatingPointError(f"PPO produced non-finite parameters: {non_finite[:8]}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
