"""IsaacLab active-vision environment for V4 spatiotemporal occupancy belief."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from codex_zbot.config import load_config
from codex_zbot.policy.features_v4 import (
    V4_ACTOR_OBSERVATION_DIM,
    V4_CRITIC_OBSERVATION_DIM,
    build_v4_actor_observation,
    build_v4_critic_observation,
)
from codex_zbot.policy.rewards import depth_consistent_visibility
from codex_zbot.policy.rewards_v4 import camera_to_oriented_boxes_clearance, normalized_v4_active_vision_reward
from codex_zbot.tasks.active_vision_env import ActiveVisionEnv, ActiveVisionEnvCfg, VISIBLE_INITIAL_POSES
from codex_zbot.tasks.scene import OBJECT_SIZES
from codex_zbot.utils.checkpoint import load_checkpoint
from codex_zbot.utils.metrics import chamfer_distance
from piaoyi_model.belief_v4 import V4BeliefLearningSystem
from piaoyi_model.config import load_drift_config
from piaoyi_model.inference_v3 import reconstruct_occupancy_v3
from piaoyi_model.model_v3 import OccupancyDriftGeneratorV3
from piaoyi_model.online_v4 import OnlineV4ObservationBuffer
from piaoyi_model.policy_adapter_v3 import v3_policy_features


@configclass
class ActiveVisionV4EnvCfg(ActiveVisionEnvCfg):
    observation_space = V4_ACTOR_OBSERVATION_DIM
    state_space = V4_CRITIC_OBSERVATION_DIM
    drift_config_path: str = str(
        Path(__file__).resolve().parents[2] / "piaoyi_model" / "configs" / "occupancy_drifting_v7_partial_visibility.json"
    )
    drift_checkpoint: str | None = None
    world_checkpoint: str | None = None
    # Surface samples measure visibility; volume samples measure occupancy completion.
    gt_surface_points: int = 2048


class ActiveVisionV4Env(DirectRLEnv):
    """PPO environment with frozen V4 generator and frozen World Model."""

    cfg: ActiveVisionV4EnvCfg
    _setup_scene = ActiveVisionEnv._setup_scene
    _camera_data = ActiveVisionEnv._camera_data
    _robot_state = ActiveVisionEnv._robot_state
    _pre_physics_step = ActiveVisionEnv._pre_physics_step
    _apply_action = ActiveVisionEnv._apply_action
    _sample_cuboid = ActiveVisionEnv._sample_cuboid
    _randomize_objects = ActiveVisionEnv._randomize_objects

    def __init__(self, cfg: ActiveVisionV4EnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.drift_checkpoint is None:
            raise ValueError("ActiveVisionV4Env requires drift_checkpoint")
        if cfg.world_checkpoint is None:
            raise ValueError("ActiveVisionV4Env requires world_checkpoint")
        self.algorithm_cfg = load_config(cfg.algorithm_config_path)
        self.drift_cfg = load_drift_config(cfg.drift_config_path)
        if self.drift_cfg.architecture != "v3_spatiotemporal":
            raise ValueError("V4 PPO requires the spatiotemporal occupancy generator")
        super().__init__(cfg, render_mode, **kwargs)

        from .scene import configure_camera_intrinsics

        configure_camera_intrinsics(self.camera)
        self.joint_ids, _ = self.robot.find_joints("joint[1-8]")
        if len(self.joint_ids) != 8:
            raise RuntimeError(f"Expected 8 zbot joints, found {len(self.joint_ids)}")
        self.drift = OccupancyDriftGeneratorV3(self.drift_cfg).to(self.device)
        load_checkpoint(cfg.drift_checkpoint, self.drift, map_location=self.device)
        self.drift.requires_grad_(False).eval()
        self.world = V4BeliefLearningSystem(self.drift_cfg, self.algorithm_cfg.world_model).to(self.device)
        load_checkpoint(cfg.world_checkpoint, self.world, map_location=self.device)
        self.world.requires_grad_(False).eval()
        self.observation_buffer = OnlineV4ObservationBuffer(self.num_envs, self.drift_cfg, self.device)
        self.inference_generator = torch.Generator(device="cpu")
        self.inference_generator.manual_seed(self.algorithm_cfg.ppo.seed + 30_003)

        batch = self.num_envs
        points = self.drift_cfg.num_points
        cells = self.drift_cfg.memory_resolution**3
        latent = self.algorithm_cfg.world_model.latent_dim
        recurrent = self.algorithm_cfg.world_model.recurrent_dim
        self.reconstruction = torch.zeros(batch, points, 3, device=self.device)
        self.tri_state_probability = torch.full((batch, cells, 3), 1.0 / 3.0, device=self.device)
        self.reconstruction_features = torch.zeros(batch, 9, device=self.device)
        self.belief_latent = torch.zeros(batch, latent, device=self.device)
        self.world_hidden = torch.zeros(batch, recurrent, device=self.device)
        self.gt_volume_cloud = torch.zeros(batch, points, 3, device=self.device)
        self.gt_surface_cloud = torch.zeros(batch, cfg.gt_surface_points, 3, device=self.device)
        self.gt_seen = torch.zeros(batch, cfg.gt_surface_points, dtype=torch.bool, device=self.device)
        self.coverage = torch.zeros(batch, device=self.device)
        self.reconstruction_error = torch.ones(batch, device=self.device)
        self.uncertainty = torch.ones(batch, device=self.device)
        self.unknown_fraction = torch.ones(batch, device=self.device)
        self.free_space_conflict = torch.zeros(batch, device=self.device)
        self.centroid_shift = torch.zeros(batch, device=self.device)
        self.valid_depth_ratio = torch.zeros(batch, device=self.device)
        self.current_surface_visibility = torch.zeros(batch, device=self.device)
        self.object_pixel_ratio = torch.zeros(batch, device=self.device)
        self.object_pixel_ratio_per_object = torch.zeros(
            batch, len(OBJECT_SIZES), device=self.device
        )
        self.minimum_object_pixel_ratio = torch.zeros(batch, device=self.device)
        self.object_missing_streak = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.has_informative_pose = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.novelty = torch.zeros(batch, device=self.device)
        self.predicted_gain = torch.zeros(batch, device=self.device)
        self.predicted_collision = torch.zeros(batch, device=self.device)
        self.collision = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.convergence_counter = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.previous_action = torch.zeros(batch, 8, device=self.device)
        self.filtered_action = torch.zeros_like(self.previous_action)
        self.safe_joint_center = self.robot.data.joint_pos[:, self.joint_ids].clone()
        self.last_reward = torch.zeros(batch, device=self.device)
        self.needs_warm_start = torch.ones(batch, dtype=torch.bool, device=self.device)
        self.object_active = torch.ones(batch, len(OBJECT_SIZES), dtype=torch.bool, device=self.device)
        self.object_position = torch.zeros(batch, len(OBJECT_SIZES), 3, device=self.device)
        self.object_yaw = torch.zeros(batch, len(OBJECT_SIZES), device=self.device)
        self.object_size_tensor = torch.tensor(OBJECT_SIZES, device=self.device)
        self.camera_clearance = torch.full((batch,), torch.inf, device=self.device)
        self.camera_proximity = torch.zeros(batch, device=self.device)
        self.camera_distance_unsafe = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.camera_workcell_unsafe = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.camera_height_unsafe = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.safety_shield_active = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.last_safe_joint_pose = self.robot.data.joint_pos[:, self.joint_ids].clone()
        self.view_shield_active = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.last_informative_joint_pose = self.robot.data.joint_pos[:, self.joint_ids].clone()

    def _observable_object_pixel_statistics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, per-active-object and minimum active-object pixel ratios.

        Semantic IDs are assigned dynamically by Isaac Sim, so they must be
        resolved through ``idToLabels`` on every rendered environment.  On
        hardware the equivalent mapping comes from the RGB instance/semantic
        segmentation front end.  Object positions and GT point clouds are not
        read by this policy-side visibility check.
        """
        semantic = self.camera.data.output["semantic_segmentation"]
        if semantic.ndim == 4 and semantic.shape[-1] == 1:
            semantic = semantic[..., 0]
        pixels = float(semantic.shape[-2] * semantic.shape[-1])
        per_object = torch.zeros(
            self.num_envs, len(OBJECT_SIZES), device=self.device, dtype=torch.float32
        )
        for env_id, sensor_info in enumerate(self.camera.data.info):
            metadata = (sensor_info or {}).get("semantic_segmentation") or {}
            id_to_labels = metadata.get("idToLabels", {})
            for raw_semantic_id, labels in id_to_labels.items():
                label_text = str(labels)
                match = re.search(r"object_(\d+)", label_text)
                if match is None:
                    continue
                object_id = int(match.group(1))
                if object_id >= len(OBJECT_SIZES) or not bool(self.object_active[env_id, object_id]):
                    continue
                per_object[env_id, object_id] = (
                    semantic[env_id] == int(raw_semantic_id)
                ).float().sum() / pixels
        active_values = torch.where(
            self.object_active,
            per_object,
            torch.full_like(per_object, torch.inf),
        )
        minimum = active_values.amin(dim=1)
        total = per_object.sum(dim=1)
        return total, per_object, minimum

    def _update_observable_object_pixels(self, env_ids: torch.Tensor | None = None) -> None:
        total, per_object, minimum = self._observable_object_pixel_statistics()
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        cfg = self.algorithm_cfg.ppo
        self.object_pixel_ratio[env_ids] = total[env_ids]
        self.object_pixel_ratio_per_object[env_ids] = per_object[env_ids]
        self.minimum_object_pixel_ratio[env_ids] = minimum[env_ids]
        missing = minimum[env_ids] < cfg.min_per_object_pixel_ratio
        self.object_missing_streak[env_ids] = torch.where(
            missing,
            self.object_missing_streak[env_ids] + 1,
            torch.zeros_like(self.object_missing_streak[env_ids]),
        )
        self.view_shield_active[env_ids] = (
            (total[env_ids] < cfg.min_object_pixel_ratio)
            | (
                self.has_informative_pose[env_ids]
                & (self.object_missing_streak[env_ids] >= cfg.max_missing_object_frames)
            )
        )

    def _update_camera_safety(self) -> None:
        """Update physical clearance and the rollback pose used by the action shield."""
        cfg = self.algorithm_cfg.ppo
        camera_position = self.camera.data.pos_w - self.scene.env_origins
        self.camera_clearance = camera_to_oriented_boxes_clearance(
            camera_position,
            self.object_position,
            self.object_size_tensor,
            self.object_yaw,
            self.object_active,
        )
        hard = cfg.camera_object_hard_distance_m
        soft = cfg.camera_object_soft_distance_m
        guard = hard + cfg.camera_safety_hysteresis_m
        distance_proximity = ((soft - self.camera_clearance) / max(soft - hard, 1.0e-6)).clamp(0.0, 1.0)
        height_proximity = (
            (cfg.camera_z_soft_min_m - camera_position[:, 2])
            / max(cfg.camera_z_soft_min_m - cfg.camera_z_hard_min_m, 1.0e-6)
        ).clamp(0.0, 1.0)
        self.camera_proximity = torch.maximum(distance_proximity, height_proximity)
        self.camera_distance_unsafe = self.camera_clearance < hard
        self.camera_workcell_unsafe = camera_position[:, 0] > cfg.camera_workcell_x_max_m
        self.camera_height_unsafe = camera_position[:, 2] < cfg.camera_z_hard_min_m
        self.safety_shield_active = (
            (self.camera_clearance < guard)
            | (camera_position[:, 0] > cfg.camera_workcell_x_max_m - cfg.camera_safety_hysteresis_m)
            | (camera_position[:, 2] < cfg.camera_z_hard_min_m + cfg.camera_safety_hysteresis_m)
        )
        safe_backup = ~self.safety_shield_active
        if safe_backup.any():
            # Preserve the last safe commanded pose.  Saving measured q would
            # bake gravity sag into the rollback target and let the camera
            # ratchet downward on every shield activation.
            commanded = getattr(self, "joint_target", self.robot.data.joint_pos[:, self.joint_ids])
            self.last_safe_joint_pose[safe_backup] = commanded[safe_backup]

    def _sample_cuboid_volume(
        self, center: torch.Tensor, size: torch.Tensor, yaw: torch.Tensor, count: int
    ) -> torch.Tensor:
        points = (torch.rand(count, 3, device=self.device) - 0.5) * size
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        x = cosine * points[:, 0] - sine * points[:, 1]
        y = sine * points[:, 0] + cosine * points[:, 1]
        points[:, 0], points[:, 1] = x.clone(), y.clone()
        return points + center

    def _rebuild_gt(self, env_ids: torch.Tensor) -> None:
        for env_id in env_ids.tolist():
            active_ids = self.object_active[env_id].nonzero(as_tuple=False).squeeze(-1).tolist()
            surface_per_object = self.cfg.gt_surface_points // len(active_ids)
            volume_per_object = self.drift_cfg.num_points // len(active_ids)
            surface_parts, volume_parts = [], []
            for object_id in active_ids:
                center = self.object_position[env_id, object_id]
                size = torch.tensor(OBJECT_SIZES[object_id], device=self.device)
                yaw = self.object_yaw[env_id, object_id]
                surface_parts.append(self._sample_cuboid(center, size, yaw, surface_per_object))
                volume_parts.append(self._sample_cuboid_volume(center, size, yaw, volume_per_object))
            for target, parts, count in (
                (self.gt_surface_cloud, surface_parts, self.cfg.gt_surface_points),
                (self.gt_volume_cloud, volume_parts, self.drift_cfg.num_points),
            ):
                cloud = torch.cat(parts)
                if cloud.shape[0] < count:
                    extra = cloud[torch.randint(0, cloud.shape[0], (count - cloud.shape[0],), device=self.device)]
                    cloud = torch.cat([cloud, extra])
                target[env_id] = cloud[:count]

    @torch.inference_mode()
    def _infer(self, env_ids: torch.Tensor, initial: bool) -> tuple[torch.Tensor, ...]:
        previous_error = self.reconstruction_error[env_ids].clone()
        previous_uncertainty = self.uncertainty[env_ids].clone()
        previous_unknown = self.unknown_fraction[env_ids].clone()
        previous_position = self.reconstruction[env_ids].clone()
        state = reconstruct_occupancy_v3(
            self.drift,
            self.observation_buffer.batch(env_ids),
            self.drift_cfg,
            self.inference_generator,
            ensemble_samples=self.algorithm_cfg.ppo.drift_ensemble_samples,
        )
        features = v3_policy_features(state, None)
        if not initial:
            features[:, 6] = previous_uncertainty - state.ensemble_uncertainty[:, 0]
            features[:, 7] = previous_unknown - state.unknown_fraction[:, 0]
            features[:, 8] = (state.position.mean(dim=1) - previous_position.mean(dim=1)).norm(dim=1)
        self.reconstruction[env_ids] = state.position
        self.tri_state_probability[env_ids] = state.tri_state_probability
        self.reconstruction_features[env_ids] = features
        self.uncertainty[env_ids] = state.ensemble_uncertainty[:, 0]
        self.unknown_fraction[env_ids] = state.unknown_fraction[:, 0]
        self.free_space_conflict[env_ids] = state.free_space_conflict[:, 0]
        self.centroid_shift[env_ids] = features[:, 8]
        self.reconstruction_error[env_ids] = chamfer_distance(state.position, self.gt_volume_cloud[env_ids])
        self.belief_latent[env_ids] = self.world.encoder(state.position, state.tri_state_probability, features)
        if initial:
            zeros = torch.zeros(len(env_ids), device=self.device)
            return zeros, zeros, zeros, zeros, zeros
        return (
            previous_error - self.reconstruction_error[env_ids],
            previous_uncertainty - self.uncertainty[env_ids],
            previous_unknown - self.unknown_fraction[env_ids],
            self.centroid_shift[env_ids],
            torch.ones(len(env_ids), device=self.device),
        )

    @torch.inference_mode()
    def _warm_start(self, env_ids: torch.Tensor) -> None:
        rgb, depth, pose, intrinsics = self._camera_data(env_ids)
        self.observation_buffer.reset(env_ids, rgb, depth, pose, intrinsics)
        self._infer(env_ids, initial=True)
        visible = depth_consistent_visibility(self.gt_surface_cloud[env_ids], depth, intrinsics, pose)
        self.gt_seen[env_ids] = visible
        self.coverage[env_ids] = visible.float().mean(dim=1)
        self.current_surface_visibility[env_ids] = visible.float().mean(dim=1)
        self._update_observable_object_pixels(env_ids)
        informative = (
            self.object_pixel_ratio[env_ids]
            >= self.algorithm_cfg.ppo.informative_object_pixel_ratio
        ) & (
            self.minimum_object_pixel_ratio[env_ids]
            >= self.algorithm_cfg.ppo.informative_per_object_pixel_ratio
        )
        if informative.any():
            informative_ids = env_ids[informative]
            self.has_informative_pose[informative_ids] = True
            self.last_informative_joint_pose[informative_ids] = self.robot.data.joint_pos[
                informative_ids
            ][:, self.joint_ids]
        valid = torch.isfinite(depth) & (depth > 0.0) & (depth <= self.drift_cfg.max_depth_m)
        self.valid_depth_ratio[env_ids] = valid.float().flatten(1).mean(dim=1)
        self.needs_warm_start[env_ids] = False

    @torch.inference_mode()
    def _update_belief(self) -> tuple[torch.Tensor, ...]:
        rgb, depth, pose, intrinsics = self._camera_data(torch.arange(self.num_envs, device=self.device))
        self.observation_buffer.append(rgb, depth, pose, intrinsics)
        previous_coverage = self.coverage.clone()
        previous_centroid_shift = self.centroid_shift.clone()
        prediction = self.world.dynamics(
            self.belief_latent, self._robot_state(), self.filtered_action, self.world_hidden
        )
        self.world_hidden = prediction.hidden
        self.predicted_gain = prediction.coverage_gain + prediction.reconstruction_gain
        self.predicted_collision = torch.sigmoid(prediction.collision_logit)

        visible = depth_consistent_visibility(self.gt_surface_cloud, depth, intrinsics, pose)
        newly_seen = visible & ~self.gt_seen
        self.novelty = newly_seen.float().sum(dim=1) / visible.float().sum(dim=1).clamp_min(1.0)
        self.gt_seen |= visible
        self.coverage = self.gt_seen.float().mean(dim=1)
        self.current_surface_visibility = visible.float().mean(dim=1)
        self._update_observable_object_pixels()
        informative = (
            self.object_pixel_ratio
            >= self.algorithm_cfg.ppo.informative_object_pixel_ratio
        ) & (
            self.minimum_object_pixel_ratio
            >= self.algorithm_cfg.ppo.informative_per_object_pixel_ratio
        )
        if informative.any():
            self.has_informative_pose[informative] = True
            self.last_informative_joint_pose[informative] = self.robot.data.joint_pos[
                informative
            ][:, self.joint_ids]
        valid = torch.isfinite(depth) & (depth > 0.0) & (depth <= self.drift_cfg.max_depth_m)
        self.valid_depth_ratio = valid.float().flatten(1).mean(dim=1)

        reconstruction_gain = torch.zeros(self.num_envs, device=self.device)
        uncertainty_gain = torch.zeros_like(reconstruction_gain)
        unknown_reduction = torch.zeros_like(reconstruction_gain)
        current_centroid_shift = previous_centroid_shift.clone()
        reconstructed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        interval = max(1, self.algorithm_cfg.ppo.reconstruction_interval)
        env_ids = ((self.episode_length_buf % interval) == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            values = self._infer(env_ids, initial=False)
            reconstruction_gain[env_ids], uncertainty_gain[env_ids], unknown_reduction[env_ids] = values[:3]
            current_centroid_shift[env_ids] = values[3]
            reconstructed[env_ids] = True
        return (
            self.coverage - previous_coverage,
            reconstruction_gain,
            uncertainty_gain,
            unknown_reduction,
            previous_centroid_shift,
            current_centroid_shift,
            reconstructed,
        )

    def _get_rewards(self) -> torch.Tensor:
        (
            coverage_gain,
            reconstruction_gain,
            uncertainty_gain,
            unknown_reduction,
            shift_previous,
            shift_current,
            reconstructed,
        ) = self._update_belief()
        limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids]
        margin = self.algorithm_cfg.ppo.joint_limit_margin_rad
        q = self.robot.data.joint_pos[:, self.joint_ids]
        joint_margin = ((q < limits[..., 0] + margin) | (q > limits[..., 1] - margin)).any(dim=1)
        terms = normalized_v4_active_vision_reward(
            self.algorithm_cfg.reward,
            coverage_gain,
            reconstruction_gain,
            uncertainty_gain,
            unknown_reduction,
            self.novelty,
            self.minimum_object_pixel_ratio,
            shift_previous,
            shift_current,
            self.filtered_action,
            self.previous_action,
            self.robot.data.joint_vel[:, self.joint_ids],
            joint_margin,
            self.collision,
            self.camera_proximity,
        )
        stable = (~reconstructed) | (shift_current < self.algorithm_cfg.reward.convergence_speed_m)
        converged = (self.coverage >= self.algorithm_cfg.reward.success_coverage) & stable
        self.convergence_counter = torch.where(converged, self.convergence_counter + 1, 0)
        success = self.convergence_counter == self.algorithm_cfg.reward.convergence_steps
        reward = terms.total + success.float() * self.algorithm_cfg.reward.success_bonus
        reward += self.reset_terminated.float() * self.algorithm_cfg.reward.unsafe_termination
        self.last_reward = reward
        self.previous_action = self.filtered_action.clone()
        self.extras["log"] = {
            "coverage": self.coverage.mean(),
            "volume_chamfer": self.reconstruction_error.mean(),
            "ensemble_uncertainty": self.uncertainty.mean(),
            "unknown_fraction": self.unknown_fraction.mean(),
            "free_space_conflict": self.free_space_conflict.mean(),
            "centroid_shift": self.centroid_shift.mean(),
            "valid_depth": self.valid_depth_ratio.mean(),
            "current_surface_visibility": self.current_surface_visibility.mean(),
            "object_pixel_ratio": self.object_pixel_ratio.mean(),
            "minimum_object_pixel_ratio": self.minimum_object_pixel_ratio.mean(),
            "missing_object_streak": self.object_missing_streak.float().mean(),
            "has_informative_pose": self.has_informative_pose.float().mean(),
            "collision_rate": self.collision.float().mean(),
            "camera_clearance": self.camera_clearance.mean(),
            "camera_proximity": self.camera_proximity.mean(),
            "camera_safety_shield": self.safety_shield_active.float().mean(),
            "camera_view_shield": self.view_shield_active.float().mean(),
            "camera_height": (self.camera.data.pos_w - self.scene.env_origins)[:, 2].mean(),
            "world_predicted_gain": self.predicted_gain.mean(),
            "world_predicted_collision": self.predicted_collision.mean(),
            "reward_coverage": terms.coverage.mean(),
            "reward_reconstruction": terms.reconstruction.mean(),
        }
        return reward

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Project incremental actions into a validated per-episode safe pose box."""
        actions = actions.clamp(-1.0, 1.0)
        alpha = self.algorithm_cfg.ppo.action_filter_alpha
        self.filtered_action = alpha * actions + (1.0 - alpha) * self.previous_action
        limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids]
        margin = self.algorithm_cfg.ppo.joint_limit_margin_rad
        excursion = self.algorithm_cfg.ppo.safe_joint_excursion_rad
        lower = torch.maximum(limits[..., 0] + margin, self.safe_joint_center - excursion)
        upper = torch.minimum(limits[..., 1] - margin, self.safe_joint_center + excursion)
        target = (
            self.robot.data.joint_pos[:, self.joint_ids]
            + self.filtered_action * self.algorithm_cfg.ppo.action_scale_rad
        )
        self.joint_target = torch.maximum(torch.minimum(target, upper), lower)
        if self.safety_shield_active.any():
            shield = self.safety_shield_active
            # A near-boundary rollback target is not sufficient when the arm
            # still has downward momentum: it can cross the hard camera-volume
            # boundary before the drive settles.  Return to the episode's
            # validated home pose instead.  This is the simulation equivalent
            # of a guarded retreat command, not a state teleport.
            self.joint_target[shield] = self.safe_joint_center[shield]
            self.filtered_action[shield] = 0.0
        if self.view_shield_active.any():
            shield = self.view_shield_active & ~self.safety_shield_active
            self.joint_target[shield] = self.last_informative_joint_pose[shield]
            self.filtered_action[shield] = 0.0

    def _apply_action(self) -> None:
        """Apply the joint target as a pure kinematic configuration update.

        The visual guidance problem depends on camera pose, not actuator
        transients.  Joint velocity is therefore identically zero and no
        inertia, damping, torque or gravity state enters the transition.
        """
        if not self.algorithm_cfg.ppo.kinematic_control:
            self.robot.set_joint_position_target(self.joint_target, joint_ids=self.joint_ids)
            return
        joint_position = self.robot.data.joint_pos.clone()
        joint_position[:, self.joint_ids] = self.joint_target
        joint_velocity = torch.zeros_like(joint_position)
        self.robot.write_joint_state_to_sim(joint_position, joint_velocity)
        self.robot.set_joint_position_target(self.joint_target, joint_ids=self.joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        warm_ids = self.needs_warm_start.nonzero(as_tuple=False).squeeze(-1)
        if len(warm_ids) > 0:
            self._warm_start(warm_ids)
        joint_position = self.robot.data.joint_pos[:, self.joint_ids]
        joint_velocity = self.robot.data.joint_vel[:, self.joint_ids]
        camera_pose = torch.cat(
            [self.camera.data.pos_w - self.scene.env_origins, self.camera.data.quat_w_world], dim=1
        )
        progress = self.episode_length_buf.float() / max(1, self.max_episode_length)
        camera_aware_risk = torch.maximum(self.predicted_collision, self.camera_proximity)
        history = torch.stack(
            [
                self.coverage,
                self.minimum_object_pixel_ratio,
                self.novelty,
                progress,
                self.predicted_gain,
                camera_aware_risk,
                self.uncertainty,
                (self.collision | self.camera_distance_unsafe | self.camera_workcell_unsafe
                 | self.camera_height_unsafe).float(),
            ],
            dim=1,
        )
        actor = build_v4_actor_observation(
            joint_position,
            joint_velocity,
            camera_pose,
            self.belief_latent,
            self.reconstruction_features,
            history,
            self.previous_action,
        )
        privileged = torch.stack(
            [
                self.coverage,
                self.reconstruction_error,
                self.uncertainty,
                self.unknown_fraction,
                (self.collision | self.camera_distance_unsafe | self.camera_workcell_unsafe
                 | self.camera_height_unsafe).float(),
                progress,
                self.minimum_object_pixel_ratio,
            ],
            dim=1,
        )
        return {"policy": actor, "critic": build_v4_critic_observation(actor, privileged)}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        force = self.contact_sensor.data.net_forces_w.norm(dim=-1).amax(dim=1)
        self.collision = force > self.cfg.collision_force_threshold_n
        self._update_camera_safety()
        limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids]
        q = self.robot.data.joint_pos[:, self.joint_ids]
        out_of_limit = ((q < limits[..., 0]) | (q > limits[..., 1])).any(dim=1)
        converged = self.convergence_counter >= self.algorithm_cfg.reward.convergence_steps
        unsafe_camera = self.camera_distance_unsafe | self.camera_workcell_unsafe | self.camera_height_unsafe
        return self.collision | unsafe_camera | out_of_limit | converged, self.episode_length_buf >= self.max_episode_length - 1

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
        self.safe_joint_center[env_ids] = initial
        self.last_safe_joint_pose[env_ids] = initial
        self.last_informative_joint_pose[env_ids] = initial
        joint_position[:, self.joint_ids] = initial
        joint_velocity = torch.zeros_like(joint_position)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_position, joint_velocity, None, env_ids)
        self.robot.set_joint_position_target(
            joint_position[:, self.joint_ids], joint_ids=self.joint_ids, env_ids=env_ids
        )
        self._randomize_objects(env_ids)
        self.gt_seen[env_ids] = False
        self.coverage[env_ids] = 0.0
        self.reconstruction_error[env_ids] = 1.0
        self.uncertainty[env_ids] = 1.0
        self.unknown_fraction[env_ids] = 1.0
        self.free_space_conflict[env_ids] = 0.0
        self.centroid_shift[env_ids] = 0.0
        self.valid_depth_ratio[env_ids] = 0.0
        self.current_surface_visibility[env_ids] = 0.0
        self.object_pixel_ratio[env_ids] = 0.0
        self.object_pixel_ratio_per_object[env_ids] = 0.0
        self.minimum_object_pixel_ratio[env_ids] = 0.0
        self.object_missing_streak[env_ids] = 0
        self.has_informative_pose[env_ids] = False
        self.novelty[env_ids] = 0.0
        self.predicted_gain[env_ids] = 0.0
        self.predicted_collision[env_ids] = 0.0
        self.collision[env_ids] = False
        self.camera_clearance[env_ids] = torch.inf
        self.camera_proximity[env_ids] = 0.0
        self.camera_distance_unsafe[env_ids] = False
        self.camera_workcell_unsafe[env_ids] = False
        self.camera_height_unsafe[env_ids] = False
        self.safety_shield_active[env_ids] = False
        self.view_shield_active[env_ids] = False
        self.convergence_counter[env_ids] = 0
        self.previous_action[env_ids] = 0.0
        self.filtered_action[env_ids] = 0.0
        self.last_reward[env_ids] = 0.0
        self.world_hidden[env_ids] = 0.0
        self.belief_latent[env_ids] = 0.0
        self.reconstruction_features[env_ids] = 0.0
        self.needs_warm_start[env_ids] = True
