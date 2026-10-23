"""Deterministic multi-episode PPO evaluation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsl_rl.runners import OnPolicyRunner
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config", required=True)
parser.add_argument("--cfm_checkpoint", required=True)
parser.add_argument("--belief_checkpoint", required=True)
parser.add_argument("--ppo_checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--output", default=None)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from codex_zbot.config import load_config
from codex_zbot.policy.rsl_rl_cfg import ActiveVisionPPORunnerCfg
from codex_zbot.tasks.active_vision_env import ActiveVisionEnv, ActiveVisionEnvCfg


def main() -> None:
    cfg = load_config(args_cli.config)
    env_cfg = ActiveVisionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = cfg.ppo.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.algorithm_config_path = str(Path(args_cli.config).resolve())
    env_cfg.cfm_checkpoint = str(Path(args_cli.cfm_checkpoint).resolve())
    env_cfg.belief_checkpoint = str(Path(args_cli.belief_checkpoint).resolve())
    raw_agent_cfg = ActiveVisionPPORunnerCfg()
    raw_agent_cfg.actor.rnn_hidden_dim = cfg.ppo.recurrent_hidden_dim
    raw_agent_cfg.critic.rnn_hidden_dim = cfg.ppo.recurrent_hidden_dim
    agent_cfg = handle_deprecated_rsl_rl_cfg(raw_agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    env = ActiveVisionEnv(env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), device=agent_cfg.device)
    runner.load(str(Path(args_cli.ppo_checkpoint).resolve()))
    policy = runner.get_inference_policy(device=agent_cfg.device)
    observation = wrapped.get_observations()

    completed = 0
    episode_returns = torch.zeros(args_cli.num_envs, device=env.device)
    records = []
    with torch.inference_mode():
        while completed < args_cli.episodes * args_cli.num_envs:
            action = policy(observation)
            observation, reward, done, info = wrapped.step(action)
            episode_returns += reward
            if done.any():
                for env_id in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    records.append(
                        {
                            "return": float(episode_returns[env_id]),
                            "coverage": float(env.coverage[env_id]),
                            "chamfer": float(env.reconstruction_error[env_id]),
                            "collision": bool(env.collision[env_id]),
                        }
                    )
                    episode_returns[env_id] = 0.0
                    completed += 1
                policy.reset(done)
    summary = {
        key: sum(float(item[key]) for item in records) / len(records)
        for key in ["return", "coverage", "chamfer", "collision"]
    }
    result = {"summary": summary, "episodes": records}
    print(json.dumps(result["summary"], indent=2))
    if args_cli.output:
        Path(args_cli.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
