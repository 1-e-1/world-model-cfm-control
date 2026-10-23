import torch

from codex_zbot.policy.rewards import depth_consistent_visibility, points_inside_image, visibility_by_object


def test_depth_visibility_rejects_occluded_point():
    # Identity world-camera convention: +X forward. Both points hit the center pixel.
    gt = torch.tensor([[[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])
    depth = torch.ones(1, 3, 3, 1)
    intrinsics = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])
    pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    visible = depth_consistent_visibility(gt, depth, intrinsics, pose, tolerance_m=0.01)
    assert visible.tolist() == [[True, False]]


def test_visibility_is_audited_per_object_instead_of_only_globally():
    visible = torch.tensor([[True, True, False, False, False, False]])
    ratios = visibility_by_object(visible, points_per_object=2)
    torch.testing.assert_close(ratios, torch.tensor([[1.0, 0.0, 0.0]]))


def test_fov_containment_respects_image_margin():
    points = torch.tensor([[[1.0, 0.0, 0.0], [1.0, -0.9, 0.0]]])
    intrinsics = torch.tensor([[[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]]])
    pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    inside = points_inside_image(points, intrinsics, pose, 100, 100, margin_px=10.0)
    assert inside.tolist() == [[True, False]]
