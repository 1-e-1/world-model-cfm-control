"""Direct IsaacLab environment for sequential CFM + active-vision PPO."""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, ContactSensor
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from codex_zbot.cfm import CFMCondition, SecondOrderCFM
from codex_zbot.config import load_config
from codex_zbot.perception.pointcloud import build_first_frame_cloud, prepare_image_condition
from codex_zbot.policy.features import ACTOR_OBSERVATION_DIM, build_actor_observation
from codex_zbot.policy.rewards import depth_consistent_visibility, normalized_active_vision_reward
from codex_zbot.utils.checkpoint import load_checkpoint
from codex_zbot.utils.metrics import chamfer_distance, flow_statistics
from codex_zbot.world_model import BeliefLearningSystem

from .scene import (
    OBJECT_SIZES,
    SCENE_OBJECT_X_RANGE,
    camera_cfg,
    configure_camera_intrinsics,
    camera_metric_intrinsics,
    contact_cfg,
    object_cfg,
    robot_cfg,
)
from .layout import SpacedLayoutConfig, sample_spaced_zero_gravity_layout


CRITIC_OBSERVATION_DIM = ACTOR_OBSERVATION_DIM + 6
VISIBLE_INITIAL_POSES = (
    (-0.581, 0.884, 0.531, -0.216, 1.060, -0.595, 1.448, -0.229),
    (-0.569, 0.865, 0.620, -0.395, 1.182, -0.506, 1.362, -0.124),
    (0.956, -0.144, 0.064, 0.164, 1.116, -0.387, 1.559, -0.533),
)


@configclass
class ActiveVisionEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 15.0
    is_finite_horizon = False
    action_space = 8
    observation_space = ACTOR_OBSERVATION_DIM
    state_space = CRITIC_OBSERVATION_DIM
    seed = 42
    num_rerenders_on_reset = 1

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0, render_interval=decimation, gravity=(0.0, 0.0, 0.0)
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=2.5, replicate_physics=True, clone_in_fabric=False
    )
    robot = robot_cfg("/World/envs/env_.*/Robot")
    camera = camera_cfg("/World/envs/env_.*/Robot/b8/Camera")
    contact_sensor = contact_cfg("/World/envs/env_.*/Robot/.*")
    objects = tuple(object_cfg(i, f"/World/envs/env_.*/object_{i}") for i in range(len(OBJECT_SIZES)))

    algorithm_config_path: str = str(Path(__file__).resolve().parents[2] / "configs" / "base.json")
    cfm_checkpoint: str | None = None
    belief_checkpoint: str | None = None
    min_objects: int = 3
    max_objects: int = 5
    scene_object_x_range_m: tuple[float, float] = SCENE_OBJECT_X_RANGE
    scene_object_y_range_m: tuple[float, float] = (-0.50, 0.50)
    min_object_xy_clearance_m: float = 0.10
    object_grounded_probability: float = 0.25
    floating_object_bottom_range_m: tuple[float, float] = (0.06, 0.38)
    collision_force_threshold_n: float = 20.0
    gt_points: int = 2048


class ActiveVisionEnv(DirectRLEnv):
    cfg: ActiveVisionEnvCfg

    def __init__(self, cfg: ActiveVisionEnvCfg, render_mode: str | None = None, **kwargs):
        self.algorithm_cfg = load_config(cfg.algorithm_config_path)
        if cfg.cfm_checkpoint is None:
            raise ValueError("ActiveVisionEnv requires a trained cfm_checkpoint")
        if self.algorithm_cfg.world_model.enabled and cfg.belief_checkpoint is None:
            raise ValueError("World Model is enabled, so a trained belief_checkpoint is required")
        self.algorithm_cfg.perception.num_points = self.algorithm_cfg.cfm.num_points
        self.algorithm_cfg.perception.observed_points = self.algorithm_cfg.cfm.observed_points
        super().__init__(cfg, render_mode, **kwargs)
        configure_camera_intrinsics(self.camera)

        self.joint_ids, _ = self.robot.find_joints("joint[1-8]")
        if len(self.joint_ids) != 8:
            raise RuntimeError(f"Expected 8 zbot joints, found {len(self.joint_ids)}")
        self.cfm = SecondOrderCFM(self.algorithm_cfg.cfm).to(self.device)
        load_checkpoint(cfg.cfm_checkpoint, self.cfm, map_location=self.device)
        self.cfm.requires_grad_(False).eval()
        self.belief: BeliefLearningSystem | None = None
        if self.algorithm_cfg.world_model.enabled:
            self.belief = BeliefLearningSystem(self.algorithm_cfg.world_model).to(self.device)
            load_checkpoint(cfg.belief_checkpoint, self.belief, map_location=self.device)
            self.belief.requires_grad_(False).eval()

        n = self.algorithm_cfg.cfm.num_points
        self.reconstruction = torch.zeros(self.num_envs, n, 3, device=self.device)
        self.flow_velocity = torch.zeros_like(self.reconstruction)
        self.flow_acceleration = torch.zeros_like(self.reconstruction)
        self.point_confidence = torch.zeros(self.num_envs, n, 1, device=self.device)
        self.particle_type = torch.zeros(self.num_envs, n, 1, device=self.device)
        self.reconstruction_latent = torch.zeros(
            self.num_envs, self.algorithm_cfg.world_model.latent_dim, device=self.device
        )
        self.world_hidden = torch.zeros(
            self.num_envs, self.algorithm_cfg.world_model.recurrent_dim, device=self.device
        )
        self.gt_cloud = torch.zeros(self.num_envs, cfg.gt_points, 3, device=self.device)
        self.gt_seen = torch.zeros(self.num_envs, cfg.gt_points, dtype=torch.bool, device=self.device)
        self.coverage = torch.zeros(self.num_envs, device=self.device)
        self.reconstruction_error = torch.ones(self.num_envs, device=self.device)
        self.uncertainty = torch.ones(self.num_envs, device=self.device)
        self.valid_depth_ratio = torch.zeros(self.num_envs, device=self.device)
        self.novelty = torch.zeros(self.num_envs, device=self.device)
        self.predicted_gain = torch.zeros(self.num_envs, device=self.device)
        self.collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.convergence_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_action = torch.zeros(self.num_envs, 8, device=self.device)
        self.filtered_action = torch.zeros_like(self.previous_action)
        self.last_reward = torch.zeros(self.num_envs, device=self.device)
        self.needs_warm_start = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_active = torch.ones(self.num_envs, len(OBJECT_SIZES), dtype=torch.bool, device=self.device)
        self.object_position = torch.zeros(self.num_envs, len(OBJECT_SIZES), 3, device=self.device)
        self.object_yaw = torch.zeros(self.num_envs, len(OBJECT_SIZES), device=self.device)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)
        self.camera = Camera(self.cfg.camera)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.objects = [RigidObject(item) for item in self.cfg.objects]
        spawn_ground_plane("/World/ground", GroundPlaneCfg(size=(4.0, 4.0)))
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["camera"] = self.camera
        self.scene.sensors["contacts"] = self.contact_sensor
        for index, item in enumerate(self.objects):
            self.scene.rigid_objects[f"object_{index}"] = item
        light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)

    def _camera_data(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = self.camera.data.output["rgb"][env_ids][..., :3]
        depth = self.camera.data.output["distance_to_image_plane"][env_ids]
        pose = torch.cat(
            [self.camera.data.pos_w[env_ids] - self.scene.env_origins[env_ids], self.camera.data.quat_w_world[env_ids]],
            dim=1,
        )
        intrinsics = camera_metric_intrinsics(self.camera)[env_ids]
        return rgb, depth, pose, intrinsics

    def _frontend(self, env_ids: torch.Tensor):
        _, depth, pose, intrinsics = self._camera_data(env_ids)
        return build_first_frame_cloud(
            depth,
            intrinsics,
            pose,
            self.algorithm_cfg.perception,
        )

    @torch.inference_mode()
    def _warm_start(self, env_ids: torch.Tensor) -> None:
        rgb, depth, pose, _ = self._camera_data(env_ids)
        frontend = self._frontend(env_ids)
        image = prepare_image_condition(
            rgb, depth, self.algorithm_cfg.cfm.image_size, self.algorithm_cfg.perception.max_depth_m
        )
        condition = CFMCondition(
            image=image,
            pose=pose,
            observed_points=frontend.observed_points,
            observed_mask=frontend.observed_mask,
            previous_points=frontend.particles,
            previous_velocity=frontend.velocity,
            confidence=frontend.confidence,
            particle_type=frontend.particle_type,
        )
        state = self.cfm(condition)
        self.reconstruction[env_ids] = state.position
        self.flow_velocity[env_ids] = state.velocity
        self.flow_acceleration[env_ids] = state.acceleration
        self.point_confidence[env_ids] = state.confidence
        self.particle_type[env_ids] = frontend.particle_type
        self.valid_depth_ratio[env_ids] = frontend.valid_depth_ratio
        visible = depth_consistent_visibility(self.gt_cloud[env_ids], depth, camera_metric_intrinsics(self.camera)[env_ids], pose)
        self.gt_seen[env_ids] = visible
        self.coverage[env_ids] = visible.float().mean(dim=1)
        self.reconstruction_error[env_ids] = chamfer_distance(state.position, self.gt_cloud[env_ids])
        self.uncertainty[env_ids] = 1.0 - state.confidence.mean(dim=(1, 2))
        if self.belief is not None:
            self.reconstruction_latent[env_ids] = self.belief.encoder(
                state.position,
                state.velocity,
                state.confidence,
                frontend.particle_type,
                acceleration=state.acceleration,
            )
        else:
            self.reconstruction_latent[env_ids] = 0.0
        self.needs_warm_start[env_ids] = False

    def _robot_state(self) -> torch.Tensor:
        joint_position = self.robot.data.joint_pos[:, self.joint_ids]
        joint_velocity = self.robot.data.joint_vel[:, self.joint_ids]
        camera_pose = torch.cat(
            [self.camera.data.pos_w - self.scene.env_origins, self.camera.data.quat_w_world], dim=1
        )
        return torch.cat([joint_position, joint_velocity, camera_pose], dim=1)

    @torch.inference_mode()
    def _update_reconstruction(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        env_ids = torch.arange(self.num_envs, device=self.device)
        rgb, depth, pose, intrinsics = self._camera_data(env_ids)
        frontend = self._frontend(env_ids)
        image = prepare_image_condition(
            rgb, depth, self.algorithm_cfg.cfm.image_size, self.algorithm_cfg.perception.max_depth_m
        )
        previous_position = self.reconstruction.clone()
        previous_speed = self.flow_velocity.norm(dim=-1).mean(dim=1)
        previous_error = self.reconstruction_error.clone()
        previous_uncertainty = self.uncertainty.clone()
        previous_coverage = self.coverage.clone()
        previous_latent = self.reconstruction_latent.clone()
        if self.belief is not None:
            prediction = self.belief.dynamics(
                previous_latent, self._robot_state(), self.filtered_action, self.world_hidden
            )
            self.world_hidden = prediction.hidden
            self.predicted_gain = prediction.coverage_gain + prediction.reconstruction_gain
        else:
            self.predicted_gain.zero_()

        condition = CFMCondition(
            image=image,
            pose=pose,
            observed_points=frontend.observed_points,
            observed_mask=frontend.observed_mask,
            previous_points=self.reconstruction,
            previous_velocity=self.flow_velocity,
            confidence=self.point_confidence,
            particle_type=self.particle_type,
        )
        state = self.cfm(condition)
        self.reconstruction = state.position
        self.flow_velocity = state.velocity
        self.flow_acceleration = state.acceleration
        self.point_confidence = state.confidence
        self.valid_depth_ratio = frontend.valid_depth_ratio
        if self.belief is not None:
            self.reconstruction_latent = self.belief.encoder(
                state.position,
                state.velocity,
                state.confidence,
                self.particle_type,
                previous_position,
                state.acceleration,
            )
        else:
            self.reconstruction_latent.zero_()
        self.reconstruction_error = chamfer_distance(state.position, self.gt_cloud)
        self.uncertainty = 1.0 - state.confidence.mean(dim=(1, 2))

        visible = depth_consistent_visibility(self.gt_cloud, depth, intrinsics, pose)
        newly_seen = visible & ~self.gt_seen
        self.novelty = newly_seen.float().sum(dim=1) / visible.float().sum(dim=1).clamp_min(1.0)
        self.gt_seen |= visible
        self.coverage = self.gt_seen.float().mean(dim=1)
        current_speed = state.velocity.norm(dim=-1).mean(dim=1)
        return (
            self.coverage - previous_coverage,
            previous_error - self.reconstruction_error,
            previous_uncertainty - self.uncertainty,
            previous_speed - current_speed,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.clamp(-1.0, 1.0)
        alpha = self.algorithm_cfg.ppo.action_filter_alpha
        self.filtered_action = alpha * actions + (1.0 - alpha) * self.previous_action
        limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids]
        margin = self.algorithm_cfg.ppo.joint_limit_margin_rad
        target = self.robot.data.joint_pos[:, self.joint_ids] + self.filtered_action * self.algorithm_cfg.ppo.action_scale_rad
        self.joint_target = torch.maximum(torch.minimum(target, limits[..., 1] - margin), limits[..., 0] + margin)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.joint_target, joint_ids=self.joint_ids)

    def _get_rewards(self) -> torch.Tensor:
        coverage_gain, reconstruction_gain, uncertainty_gain, speed_gain = self._update_reconstruction()
        current_speed = self.flow_velocity.norm(dim=-1).mean(dim=1)
        previous_speed = current_speed + speed_gain
        limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids]
        margin = self.algorithm_cfg.ppo.joint_limit_margin_rad
        q = self.robot.data.joint_pos[:, self.joint_ids]
        joint_margin = ((q < limits[..., 0] + margin) | (q > limits[..., 1] - margin)).any(dim=1)
        terms = normalized_active_vision_reward(
            self.algorithm_cfg.reward,
            coverage_gain,
            reconstruction_gain,
            uncertainty_gain,
            self.novelty,
            self.valid_depth_ratio,
            previous_speed,
            current_speed,
            self.filtered_action,
            self.previous_action,
            self.robot.data.joint_vel[:, self.joint_ids],
            joint_margin,
            self.collision,
        )
        converged = (self.coverage >= self.algorithm_cfg.reward.success_coverage) & (
            current_speed < self.algorithm_cfg.reward.convergence_speed_m
        )
        self.convergence_counter = torch.where(converged, self.convergence_counter + 1, 0)
        success = self.convergence_counter == self.algorithm_cfg.reward.convergence_steps
        reward = terms.total + success.float() * self.algorithm_cfg.reward.success_bonus
        reward += self.reset_terminated.float() * self.algorithm_cfg.reward.unsafe_termination
        self.last_reward = reward
        self.previous_action = self.filtered_action.clone()
        self.extras["log"] = {
            "coverage": self.coverage.mean(),
            "chamfer": self.reconstruction_error.mean(),
            "flow_speed": current_speed.mean(),
            "valid_depth": self.valid_depth_ratio.mean(),
            "collision_rate": self.collision.float().mean(),
            "reward_coverage": terms.coverage.mean(),
            "reward_reconstruction": terms.reconstruction.mean(),
        }
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        warm_ids = self.needs_warm_start.nonzero(as_tuple=False).squeeze(-1)
        if len(warm_ids) > 0:
            self._warm_start(warm_ids)
        joint_position = self.robot.data.joint_pos[:, self.joint_ids]
        joint_velocity = self.robot.data.joint_vel[:, self.joint_ids]
        camera_pose = torch.cat(
            [self.camera.data.pos_w - self.scene.env_origins, self.camera.data.quat_w_world], dim=1
        )
        stats = flow_statistics(
            self.reconstruction,
            self.reconstruction - self.flow_velocity * self.step_dt,
            self.flow_velocity,
            self.flow_acceleration,
        )
        progress = self.episode_length_buf.float() / max(1, self.max_episode_length)
        history = torch.stack(
            [
                self.coverage,
                self.valid_depth_ratio,
                self.novelty,
                1.0 - self.novelty,
                progress,
                self.predicted_gain,
                self.uncertainty,
                self.collision.float(),
            ],
            dim=1,
        )
        policy = build_actor_observation(
            joint_position,
            joint_velocity,
            camera_pose,
            self.reconstruction_latent,
            stats,
            history,
            self.previous_action,
        )
        critic_extra = torch.stack(
            [
                self.coverage,
                self.reconstruction_error,
                self.uncertainty,
                self.collision.float(),
                progress,
                self.valid_depth_ratio,
            ],
            dim=1,
        )
        return {"policy": policy, "critic": torch.cat([policy, critic_extra], dim=1)}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        force = self.contact_sensor.data.net_forces_w.norm(dim=-1).amax(dim=1)
        self.collision = force > self.cfg.collision_force_threshold_n
        limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids]
        q = self.robot.data.joint_pos[:, self.joint_ids]
        out_of_limit = ((q < limits[..., 0]) | (q > limits[..., 1])).any(dim=1)
        converged = self.convergence_counter >= self.algorithm_cfg.reward.convergence_steps
        terminated = self.collision | out_of_limit | converged
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _sample_cuboid(self, center: torch.Tensor, size: torch.Tensor, yaw: torch.Tensor, count: int) -> torch.Tensor:
        face = torch.randint(0, 6, (count,), device=self.device)
        points = (torch.rand(count, 3, device=self.device) * 2.0 - 1.0) * (size / 2.0)
        axis = face // 2
        sign = torch.where(face % 2 == 0, 1.0, -1.0)
        points[torch.arange(count, device=self.device), axis] = sign * size[axis] / 2.0
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        x = cosine * points[:, 0] - sine * points[:, 1]
        y = sine * points[:, 0] + cosine * points[:, 1]
        points[:, 0], points[:, 1] = x, y
        return points + center

    def _rebuild_gt(self, env_ids: torch.Tensor) -> None:
        ground_count = self.cfg.gt_points // 4
        for env_id in env_ids.tolist():
            active_ids = self.object_active[env_id].nonzero(as_tuple=False).squeeze(-1).tolist()
            per_object = (self.cfg.gt_points - ground_count) // len(active_ids)
            ground = torch.rand(ground_count, 3, device=self.device)
            ground[:, 0] = 0.15 + (self.cfg.scene_object_x_range_m[1] + 0.08 - 0.15) * ground[:, 0]
            ground[:, 1] = (
                self.cfg.scene_object_y_range_m[0] - 0.08
                + (self.cfg.scene_object_y_range_m[1] - self.cfg.scene_object_y_range_m[0] + 0.16)
                * ground[:, 1]
            )
            ground[:, 2] = 0.0
            parts = [ground]
            for object_id in active_ids:
                parts.append(
                    self._sample_cuboid(
                        self.object_position[env_id, object_id],
                        torch.tensor(OBJECT_SIZES[object_id], device=self.device),
                        self.object_yaw[env_id, object_id],
                        per_object,
                    )
                )
            cloud = torch.cat(parts)
            if cloud.shape[0] < self.cfg.gt_points:
                extra = cloud[torch.randint(0, cloud.shape[0], (self.cfg.gt_points - cloud.shape[0],), device=self.device)]
                cloud = torch.cat([cloud, extra])
            self.gt_cloud[env_id] = cloud[: self.cfg.gt_points]

    def _randomize_objects(self, env_ids: torch.Tensor) -> None:
        for env_id in env_ids.tolist():
            count = int(torch.randint(self.cfg.min_objects, self.cfg.max_objects + 1, ()).item())
            permutation = torch.randperm(len(self.objects), device=self.device)
            self.object_active[env_id] = False
            self.object_active[env_id, permutation[:count]] = True
            active_ids = permutation[:count].detach().cpu().tolist()
            layout_seed = int(torch.randint(0, 2**31 - 1, ()).item())
            centers_np, yaws_np, _ = sample_spaced_zero_gravity_layout(
                np.asarray([OBJECT_SIZES[index] for index in active_ids], dtype=np.float32),
                np.random.default_rng(layout_seed),
                SpacedLayoutConfig(
                    x_range_m=self.cfg.scene_object_x_range_m,
                    y_range_m=self.cfg.scene_object_y_range_m,
                    min_surface_clearance_m=self.cfg.min_object_xy_clearance_m,
                    grounded_probability=self.cfg.object_grounded_probability,
                    floating_bottom_range_m=self.cfg.floating_object_bottom_range_m,
                ),
            )
            layout_by_object = {
                object_id: (centers_np[slot], yaws_np[slot])
                for slot, object_id in enumerate(active_ids)
            }
            for object_id, obj in enumerate(self.objects):
                active = bool(self.object_active[env_id, object_id])
                if active:
                    center_np, yaw_np = layout_by_object[object_id]
                    local_position = torch.as_tensor(center_np, device=self.device)
                    yaw = torch.as_tensor(yaw_np, device=self.device)
                else:
                    local_position = torch.tensor([0.45, 0.0, -2.0], device=self.device)
                    yaw = torch.zeros((), device=self.device)
                self.object_position[env_id, object_id] = local_position
                self.object_yaw[env_id, object_id] = yaw
                pose = torch.zeros(1, 7, device=self.device)
                pose[0, :3] = local_position + self.scene.env_origins[env_id]
                pose[0, 3] = torch.cos(yaw / 2.0)
                pose[0, 6] = torch.sin(yaw / 2.0)
                ids = torch.tensor([env_id], device=self.device)
                obj.write_root_pose_to_sim(pose, ids)
                obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device), ids)
        self._rebuild_gt(env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        joint_position = self.robot.data.default_joint_pos[env_ids].clone()
        pose_bank = torch.tensor(VISIBLE_INITIAL_POSES, device=self.device, dtype=joint_position.dtype)
        selected = torch.randint(0, pose_bank.shape[0], (len(env_ids),), device=self.device)
        initial = pose_bank[selected] + torch.randn(len(env_ids), 8, device=self.device) * 0.02
        limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.joint_ids]
        initial = torch.maximum(torch.minimum(initial, limits[..., 1] - 0.1), limits[..., 0] + 0.1)
        joint_position[:, self.joint_ids] = initial
        joint_velocity = torch.zeros_like(joint_position)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_position, joint_velocity, None, env_ids)
        self.robot.set_joint_position_target(joint_position[:, self.joint_ids], joint_ids=self.joint_ids, env_ids=env_ids)
        self._randomize_objects(env_ids)

        self.gt_seen[env_ids] = False
        self.coverage[env_ids] = 0.0
        self.reconstruction_error[env_ids] = 1.0
        self.uncertainty[env_ids] = 1.0
        self.valid_depth_ratio[env_ids] = 0.0
        self.novelty[env_ids] = 0.0
        self.predicted_gain[env_ids] = 0.0
        self.collision[env_ids] = False
        self.convergence_counter[env_ids] = 0
        self.previous_action[env_ids] = 0.0
        self.filtered_action[env_ids] = 0.0
        self.last_reward[env_ids] = 0.0
        self.world_hidden[env_ids] = 0.0
        self.needs_warm_start[env_ids] = True
