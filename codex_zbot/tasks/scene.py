"""IsaacLab scene assets for the Codex active-vision task."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from codex_zbot.perception.pointcloud import renderer_intrinsics_to_pixel_indices

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg

from .scene_constants import (
    CAMERA_HEIGHT,
    CAMERA_HORIZONTAL_FOV_DEG,
    CAMERA_WIDTH,
    OBJECT_COLORS,
    OBJECT_SIZES,
    ROBOT_BASE_X_M,
    SCENE_OBJECT_X_RANGE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = str(PROJECT_ROOT / "zbot_8s_new.usd")
def robot_cfg(prim_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # The physical arm's joint servos provide gravity
                # compensation.  Leaving free gravity enabled here made the
                # camera sag even while the commanded joint target was held,
                # which incorrectly coupled simulator actuator limits into the
                # visual-servo policy.
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(ROBOT_BASE_X_M, 0.0, 0.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-8]"],
                stiffness=1200.0,
                damping=60.0,
                velocity_limit_sim=2.0,
            )
        },
    )


def camera_cfg(prim_path: str, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        spawn=None,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        width=width,
        height=height,
        colorize_semantic_segmentation=False,
        depth_clipping_behavior="zero",
        update_latest_camera_pose=True,
    )


def pinhole_intrinsics(
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
    horizontal_fov_deg: float = CAMERA_HORIZONTAL_FOV_DEG,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build centered square-pixel intrinsics from the requested horizontal FOV."""
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")
    if not 1.0 < horizontal_fov_deg < 179.0:
        raise ValueError("horizontal_fov_deg must be in (1, 179)")
    focal_px = 0.5 * width / math.tan(math.radians(horizontal_fov_deg) * 0.5)
    return torch.tensor(
        [[focal_px, 0.0, width * 0.5], [0.0, focal_px, height * 0.5], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def configure_camera_intrinsics(camera: Camera, horizontal_fov_deg: float = CAMERA_HORIZONTAL_FOV_DEG) -> torch.Tensor:
    """Override the camera embedded in the robot USD with the active-vision lens profile."""
    count = camera.data.intrinsic_matrices.shape[0]
    matrix = pinhole_intrinsics(
        camera.cfg.width,
        camera.cfg.height,
        horizontal_fov_deg,
        device=camera.device,
        dtype=camera.data.intrinsic_matrices.dtype,
    )
    matrices = matrix.unsqueeze(0).expand(count, -1, -1).clone()
    # The default IsaacLab setter uses a sub-unit USD focal length. In this
    # renderer it produced ~90-degree images despite a 110-degree K readback.
    # A conventional focal/aperture scale preserves K without that mismatch:
    # scripts/audit_camera_lens.py measures plane MAE 37.26 -> 3.51 mm.
    # Newly collected images need their own validation; this does NOT repair
    # old depth files or authorize reusing their calibration / PPO weights.
    camera.set_intrinsic_matrices(matrices, focal_length=24.0)
    return camera_metric_intrinsics(camera)


def camera_metric_intrinsics(camera):
    """K for integer-index depth arrays, without changing the physical lens."""
    return renderer_intrinsics_to_pixel_indices(camera.data.intrinsic_matrices)


def contact_cfg(prim_path: str) -> ContactSensorCfg:
    return ContactSensorCfg(prim_path=prim_path, history_length=1, track_air_time=False)


def object_cfg(index: int, prim_path: str, size: tuple[float, float, float] | None = None,
               shape: str = "cuboid") -> RigidObjectCfg:
    object_size = tuple(size) if size is not None else OBJECT_SIZES[index]
    common=dict(
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=OBJECT_COLORS[index%len(OBJECT_COLORS)]),
        semantic_tags=[("class",f"object_{index}")],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    if shape=="cuboid":
        spawn=sim_utils.CuboidCfg(size=object_size,**common)
    elif shape=="sphere":
        if max(object_size)-min(object_size)>1e-6:
            raise ValueError("sphere sizes must be equal diameters")
        spawn=sim_utils.SphereCfg(radius=object_size[0]*.5,**common)
    elif shape=="cylinder":
        if abs(object_size[0]-object_size[1])>1e-6:
            raise ValueError("cylinder x/y sizes must equal diameter")
        spawn=sim_utils.CylinderCfg(radius=object_size[0]*.5,height=object_size[2],axis="Z",**common)
    else:
        raise ValueError(f"unsupported object shape: {shape}")
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, object_size[2] / 2.0)),
    )
