import torch

from codex_zbot.config import PerceptionConfig
from codex_zbot.perception.pointcloud import backproject_depth_world, build_first_frame_cloud, prepare_image_condition


def test_identity_camera_center_pixel_points_forward_x():
    depth = torch.zeros(1, 3, 3, 1)
    depth[0, 1, 1, 0] = 1.0
    intrinsics = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])
    pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    points, valid = backproject_depth_world(depth, intrinsics, pose, 2.0)
    center = points[0, 4]
    torch.testing.assert_close(center, torch.tensor([1.0, 0.0, 0.0]))
    assert valid.sum() == 1


def test_first_frame_contains_measured_and_latent_particles():
    depth = torch.ones(1, 8, 8, 1) * 0.5
    intrinsics = torch.tensor([[[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]]])
    pose = torch.tensor([[0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]])
    cfg = PerceptionConfig(
        num_points=32,
        observed_points=16,
        voxel_size_m=0.0,
        outlier_radius_m=0.0,
        workspace_min=(0.0, -1.0, -1.0),
        workspace_max=(1.0, 1.0, 1.0),
    )
    result = build_first_frame_cloud(depth, intrinsics, pose, cfg, generator=torch.Generator().manual_seed(0))
    assert result.particles.shape == (1, 32, 3)
    assert result.observed_points.shape == (1, 16, 3)
    assert result.particle_type.sum() > 0
    assert result.confidence.sum() > 0
    assert torch.count_nonzero(result.velocity) == 0


def test_image_condition_has_five_channels():
    rgb = torch.zeros(2, 8, 8, 3, dtype=torch.uint8)
    depth = torch.ones(2, 8, 8, 1)
    image = prepare_image_condition(rgb, depth, 4, 2.0)
    assert image.shape == (2, 5, 4, 4)

