"""Deterministic V4 PPO evaluation with occupancy and safety metrics."""
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
parser.add_argument("--config", default=str(ROOT / "configs/formal_v4.json"))
parser.add_argument("--drift_config", default=str(ROOT / "piaoyi_model/configs/occupancy_drifting_v7_partial_visibility.json"))
parser.add_argument("--drift_checkpoint", required=True)
parser.add_argument("--world_checkpoint", required=True)
parser.add_argument("--ppo_checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--output", required=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from codex_zbot.config import load_config
from codex_zbot.policy.rsl_rl_cfg import ActiveVisionPPORunnerCfg
from codex_zbot.tasks.active_vision_v4_env import ActiveVisionV4Env, ActiveVisionV4EnvCfg


def main() -> None:
    cfg = load_config(args_cli.config)
    env_cfg = ActiveVisionV4EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = cfg.ppo.seed + 1000
    env_cfg.sim.device = args_cli.device
    env_cfg.algorithm_config_path = str(Path(args_cli.config).resolve())
    env_cfg.drift_config_path = str(Path(args_cli.drift_config).resolve())
    env_cfg.drift_checkpoint = str(Path(args_cli.drift_checkpoint).resolve())
    env_cfg.world_checkpoint = str(Path(args_cli.world_checkpoint).resolve())
    env_cfg.decimation = cfg.ppo.decimation
    env_cfg.episode_length_s = cfg.ppo.episode_length_s
    env_cfg.sim.dt = cfg.ppo.physics_dt
    env_cfg.sim.render_interval = env_cfg.decimation
    raw = ActiveVisionPPORunnerCfg()
    raw.actor.rnn_hidden_dim = cfg.ppo.recurrent_hidden_dim
    raw.critic.rnn_hidden_dim = cfg.ppo.recurrent_hidden_dim
    agent_cfg = handle_deprecated_rsl_rl_cfg(raw, importlib.metadata.version("rsl-rl-lib"))
    env = ActiveVisionV4Env(env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), device=agent_cfg.device)
    runner.load(str(Path(args_cli.ppo_checkpoint).resolve()))
    policy = runner.get_inference_policy(device=agent_cfg.device)
    observation = wrapped.get_observations()
    returns = torch.zeros(args_cli.num_envs, device=env.device)
    initial_coverage = env.coverage.clone()
    initial_chamfer = env.reconstruction_error.clone()
    minimum_clearance = torch.full((args_cli.num_envs,), torch.inf, device=env.device)
    shield_steps = torch.zeros(args_cli.num_envs, device=env.device)
    maximum_camera_x = torch.full((args_cli.num_envs,), -torch.inf, device=env.device)
    minimum_camera_z = torch.full((args_cli.num_envs,), torch.inf, device=env.device)
    minimum_surface_visibility = torch.full((args_cli.num_envs,), torch.inf, device=env.device)
    minimum_object_pixel_ratio = torch.full((args_cli.num_envs,), torch.inf, device=env.device)
    minimum_per_object_pixel_ratio = torch.full((args_cli.num_envs,), torch.inf, device=env.device)
    view_shield_steps = torch.zeros(args_cli.num_envs, device=env.device)
    uninformative_steps = torch.zeros(args_cli.num_envs, device=env.device)
    records = []
    target = args_cli.episodes * args_cli.num_envs
    with torch.inference_mode():
        while len(records) < target:
            env._update_camera_safety()
            camera_x = (env.camera.data.pos_w - env.scene.env_origins)[:, 0]
            minimum_clearance = torch.minimum(minimum_clearance, env.camera_clearance)
            maximum_camera_x = torch.maximum(maximum_camera_x, camera_x)
            camera_z = (env.camera.data.pos_w - env.scene.env_origins)[:, 2]
            minimum_camera_z = torch.minimum(minimum_camera_z, camera_z)
            shield_steps += env.safety_shield_active.float()
            minimum_surface_visibility = torch.minimum(
                minimum_surface_visibility, env.current_surface_visibility
            )
            minimum_object_pixel_ratio = torch.minimum(
                minimum_object_pixel_ratio, env.object_pixel_ratio
            )
            minimum_per_object_pixel_ratio = torch.minimum(
                minimum_per_object_pixel_ratio, env.minimum_object_pixel_ratio
            )
            view_shield_steps += env.view_shield_active.float()
            uninformative_steps += env.view_shield_active.float()
            # DirectRLEnv auto-resets completed environments inside step().
            # Cache the old episode state before stepping; reading env fields
            # after done would silently report the next episode's reset state.
            terminal_coverage = env.coverage.clone()
            terminal_chamfer = env.reconstruction_error.clone()
            terminal_unknown = env.unknown_fraction.clone()
            terminal_conflict = env.free_space_conflict.clone()
            terminal_length = env.episode_length_buf.clone()
            observation, reward, done, info = wrapped.step(policy(observation))
            returns += reward
            if done.any():
                time_outs = info.get("time_outs", torch.zeros_like(done, dtype=torch.bool)).bool()
                for env_id in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    if len(records) >= target:
                        break
                    unsafe = bool(
                        not time_outs[env_id]
                        and terminal_coverage[env_id] < cfg.reward.success_coverage
                        # Timeout is decided after DirectRLEnv increments the
                        # buffer.  The pre-step cached value is therefore
                        # max_episode_length-2 for a normal horizon timeout.
                        and terminal_length[env_id] < env.max_episode_length - 2
                    )
                    records.append(
                        {
                            "return": float(returns[env_id]),
                            "coverage": float(terminal_coverage[env_id]),
                            "coverage_gain": float(terminal_coverage[env_id] - initial_coverage[env_id]),
                            "volume_chamfer": float(terminal_chamfer[env_id]),
                            "chamfer_gain": float(initial_chamfer[env_id] - terminal_chamfer[env_id]),
                            "unknown_fraction": float(terminal_unknown[env_id]),
                            "free_space_conflict": float(terminal_conflict[env_id]),
                            "minimum_camera_clearance": float(minimum_clearance[env_id]),
                            "maximum_camera_x": float(maximum_camera_x[env_id]),
                            "minimum_camera_z": float(minimum_camera_z[env_id]),
                            "camera_shield_fraction": float(
                                shield_steps[env_id] / terminal_length[env_id].clamp_min(1)
                            ),
                            "minimum_surface_visibility": float(minimum_surface_visibility[env_id]),
                            "minimum_object_pixel_ratio": float(minimum_object_pixel_ratio[env_id]),
                            "minimum_per_object_pixel_ratio": float(
                                minimum_per_object_pixel_ratio[env_id]
                            ),
                            "view_shield_fraction": float(
                                view_shield_steps[env_id] / terminal_length[env_id].clamp_min(1)
                            ),
                            "uninformative_frame_fraction": float(
                                uninformative_steps[env_id] / terminal_length[env_id].clamp_min(1)
                            ),
                            "collision": unsafe,
                            "unsafe_termination": unsafe,
                        }
                    )
                    returns[env_id] = 0.0
                    # The environment has already auto-reset at this point.
                    initial_coverage[env_id] = env.coverage[env_id]
                    initial_chamfer[env_id] = env.reconstruction_error[env_id]
                    minimum_clearance[env_id] = torch.inf
                    maximum_camera_x[env_id] = -torch.inf
                    minimum_camera_z[env_id] = torch.inf
                    shield_steps[env_id] = 0.0
                    minimum_surface_visibility[env_id] = torch.inf
                    minimum_object_pixel_ratio[env_id] = torch.inf
                    minimum_per_object_pixel_ratio[env_id] = torch.inf
                    view_shield_steps[env_id] = 0.0
                    uninformative_steps[env_id] = 0.0
                policy.reset(done)
    keys = list(records[0])
    summary = {key: sum(float(item[key]) for item in records) / len(records) for key in keys}
    result = {
        "summary": summary,
        "episodes": records,
        "seeds": {"environment": env_cfg.seed, "model": cfg.ppo.seed},
        "checkpoint": str(Path(args_cli.ppo_checkpoint).resolve()),
    }
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
