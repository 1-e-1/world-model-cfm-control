"""Pure geometry utilities for deterministic zero-gravity scene layouts.

This module intentionally has no IsaacLab dependency so placement constraints
can be unit-tested before launching the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .shape_geometry import shape_half_extent


def sample_cuboid_sizes(
    count: int,
    rng: np.random.Generator,
    minimum_m: tuple[float, float, float] = (0.06, 0.06, 0.08),
    maximum_m: tuple[float, float, float] = (0.16, 0.16, 0.24),
) -> np.ndarray:
    """Independently sample every cuboid dimension from continuous ranges."""
    minimum = np.asarray(minimum_m, dtype=np.float32)
    maximum = np.asarray(maximum_m, dtype=np.float32)
    if count <= 0 or not (maximum > minimum).all():
        raise ValueError("count must be positive and every size range must be increasing")
    return rng.uniform(minimum, maximum, size=(count, 3)).astype(np.float32)


@dataclass(frozen=True)
class SpacedLayoutConfig:
    x_range_m: tuple[float, float] = (0.54, 0.94)
    y_range_m: tuple[float, float] = (-0.50, 0.50)
    min_surface_clearance_m: float = 0.10
    grounded_probability: float = 0.25
    floating_bottom_range_m: tuple[float, float] = (0.06, 0.38)
    max_roll_pitch_rad: float = 0.0
    attempts_per_restart: int = 512
    restarts: int = 128
    ensure_floating_object: bool = True
    max_nearest_surface_clearance_m: float | None = None


def oriented_xy_aabb_half_extent(size: np.ndarray, yaw: float) -> np.ndarray:
    """Conservative XY half extent of a yaw-rotated cuboid."""
    cosine, sine = abs(np.cos(yaw)), abs(np.sin(yaw))
    return np.asarray(
        [
            0.5 * (cosine * size[0] + sine * size[1]),
            0.5 * (sine * size[0] + cosine * size[1]),
        ],
        dtype=np.float32,
    )


def rotation_matrix_xyz(euler_xyz: np.ndarray) -> np.ndarray:
    """Return the active Rz @ Ry @ Rx rotation for XYZ Euler angles."""
    roll, pitch, yaw = (float(value) for value in euler_xyz)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def oriented_aabb_half_extent(size: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
    return np.abs(rotation_matrix_xyz(euler_xyz)) @ (0.5 * np.asarray(size, dtype=np.float32))


def quaternion_wxyz_from_euler_xyz(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles to a normalized wxyz quaternion."""
    roll, pitch, yaw = 0.5 * np.asarray(euler_xyz, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    quaternion = np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )
    return quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-8)


def xy_aabb_clearance(
    center_a: np.ndarray,
    half_extent_a: np.ndarray,
    center_b: np.ndarray,
    half_extent_b: np.ndarray,
) -> float:
    """Euclidean separation between conservative axis-aligned XY footprints."""
    separation = np.maximum(
        np.abs(center_a[:2] - center_b[:2]) - half_extent_a - half_extent_b,
        0.0,
    )
    return float(np.linalg.norm(separation))


def aabb_surface_clearance(
    center_a: np.ndarray,
    half_extent_a: np.ndarray,
    center_b: np.ndarray,
    half_extent_b: np.ndarray,
) -> float:
    """Conservative 3-D surface clearance between two oriented-box AABBs."""
    separation = np.maximum(
        np.abs(center_a - center_b) - half_extent_a - half_extent_b,
        0.0,
    )
    return float(np.linalg.norm(separation))


def sample_spaced_zero_gravity_layout(
    sizes: np.ndarray,
    rng: np.random.Generator,
    config: SpacedLayoutConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample separated yawed cuboids, with a guaranteed floating object.

    Returns centers ``(N,3)``, yaws ``(N,)`` and grounded flags ``(N,)`` in
    the original input order.  Clearance is enforced in XY even when two
    objects are separated vertically, keeping them visually distinguishable.
    """
    centers, euler_xyz, grounded = sample_spaced_zero_gravity_layout_6d(sizes, rng, config)
    return centers, euler_xyz[:, 2], grounded


def sample_spaced_zero_gravity_layout_6d(
    sizes: np.ndarray,
    rng: np.random.Generator,
    config: SpacedLayoutConfig | None = None,
    shape_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample separated primitives with continuous position and 3-D orientation.

    The returned orientation is XYZ Euler in radians. Ground contact is based
    on the true primitive support extent, so tilted primitives do not
    intersect the plane. Omitted shape IDs retain the legacy cuboid path.
    """
    cfg = config or SpacedLayoutConfig()
    sizes = np.asarray(sizes, dtype=np.float32)
    if sizes.ndim != 2 or sizes.shape[1] != 3 or len(sizes) == 0:
        raise ValueError("sizes must have shape (N,3) with N > 0")
    shape_ids=np.zeros(len(sizes),dtype=np.int32) if shape_ids is None else np.asarray(shape_ids)
    if shape_ids.shape != (len(sizes),) or not np.isin(shape_ids,[0,1,2]).all():
        raise ValueError("invalid layout shape IDs")
    if cfg.x_range_m[0] >= cfg.x_range_m[1] or cfg.y_range_m[0] >= cfg.y_range_m[1]:
        raise ValueError("layout ranges must be strictly increasing")
    if cfg.min_surface_clearance_m < 0:
        raise ValueError("min_surface_clearance_m must be non-negative")
    if cfg.max_nearest_surface_clearance_m is not None and cfg.max_nearest_surface_clearance_m < cfg.min_surface_clearance_m:
        raise ValueError("nearest-neighbor maximum must not be below the minimum clearance")
    if not 0.0 <= cfg.grounded_probability <= 1.0:
        raise ValueError("grounded_probability must be in [0,1]")

    # Largest footprints first makes rejection sampling reliable at five
    # objects while output is restored to the caller's original object order.
    order = np.argsort(-(sizes[:, 0] * sizes[:, 1]))
    for _ in range(cfg.restarts):
        centers = np.zeros((len(sizes), 3), dtype=np.float32)
        euler_xyz = np.zeros((len(sizes), 3), dtype=np.float32)
        half_extents = np.zeros((len(sizes), 3), dtype=np.float32)
        grounded = rng.random(len(sizes)) < cfg.grounded_probability
        if cfg.ensure_floating_object and len(sizes) > 1 and grounded.all():
            grounded[int(rng.integers(len(sizes)))] = False
        placed: list[int] = []
        success = True
        for object_id in order.tolist():
            size = sizes[object_id]
            accepted = False
            for _ in range(cfg.attempts_per_restart):
                euler = np.asarray(
                    [
                        rng.uniform(-cfg.max_roll_pitch_rad, cfg.max_roll_pitch_rad),
                        rng.uniform(-cfg.max_roll_pitch_rad, cfg.max_roll_pitch_rad),
                        rng.uniform(-np.pi, np.pi),
                    ],
                    dtype=np.float32,
                )
                half = shape_half_extent(size,rotation_matrix_xyz(euler),shape_ids[object_id])
                x_low, x_high = cfg.x_range_m[0] + half[0], cfg.x_range_m[1] - half[0]
                y_low, y_high = cfg.y_range_m[0] + half[1], cfg.y_range_m[1] - half[1]
                if x_low >= x_high or y_low >= y_high:
                    raise ValueError("layout range is too small for an object")
                center_xy = np.asarray(
                    [rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)], dtype=np.float32
                )
                bottom = (
                    0.0
                    if grounded[object_id]
                    else float(rng.uniform(*cfg.floating_bottom_range_m))
                )
                center = np.asarray([center_xy[0], center_xy[1], bottom + half[2]], dtype=np.float32)
                if all(
                    aabb_surface_clearance(center, half, centers[other], half_extents[other])
                    >= cfg.min_surface_clearance_m
                    for other in placed
                ):
                    centers[object_id] = center
                    euler_xyz[object_id] = euler
                    half_extents[object_id] = half
                    placed.append(object_id)
                    accepted = True
                    break
            if not accepted:
                success = False
                break
        if not success:
            continue

        if cfg.max_nearest_surface_clearance_m is not None and len(sizes)>1:
            # Each convex primitive contains its centered inscribed sphere.
            # Distance between those spheres bounds the true surface gap
            # from above. AABB clearance above bounds it from below.
            radius=sizes.min(axis=1)*.5
            gap_upper=np.maximum(np.linalg.norm(centers[:,None]-centers[None,:],axis=2)
                                 -radius[:,None]-radius[None,:],0)
            np.fill_diagonal(gap_upper,np.inf)
            if (gap_upper.min(axis=1)>cfg.max_nearest_surface_clearance_m).any():
                continue

        return centers, euler_xyz, grounded
    raise RuntimeError(
        f"unable to place {len(sizes)} objects with {cfg.min_surface_clearance_m:.3f} m clearance"
    )
