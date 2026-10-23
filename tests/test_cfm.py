import torch

from codex_zbot.cfm import CFMCondition, SecondOrderCFM, compute_cfm_loss
from codex_zbot.config import load_config


def make_condition(cfg, batch=2):
    n, m, size = cfg.cfm.num_points, cfg.cfm.observed_points, cfg.cfm.image_size
    return CFMCondition(
        image=torch.rand(batch, cfg.cfm.image_channels, size, size),
        pose=torch.nn.functional.normalize(torch.rand(batch, 7), dim=-1),
        observed_points=torch.rand(batch, m, 3),
        observed_mask=torch.ones(batch, m, dtype=torch.bool),
        previous_points=torch.rand(batch, n, 3),
        previous_velocity=torch.zeros(batch, n, 3),
        confidence=torch.rand(batch, n, 1),
        particle_type=torch.randint(0, 2, (batch, n, 1)).float(),
    )


def test_cfm_forward_and_loss_are_finite():
    cfg = load_config("configs/smoke.json")
    model = SecondOrderCFM(cfg.cfm)
    condition = make_condition(cfg)
    state = model(condition)
    assert state.position.shape == condition.previous_points.shape
    assert state.velocity.shape == condition.previous_velocity.shape
    target = torch.rand(2, cfg.cfm.num_points, 3)
    loss = compute_cfm_loss(model, condition, target, cfg.cfm_loss, torch.Generator().manual_seed(2))
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_zero_observation_mask_does_not_create_nan():
    cfg = load_config("configs/smoke.json")
    model = SecondOrderCFM(cfg.cfm)
    condition = make_condition(cfg, batch=1)
    condition.observed_mask.zero_()
    state = model(condition)
    assert torch.isfinite(state.position).all()


def test_first_order_ablation_removes_velocity_dynamics():
    cfg = load_config("configs/smoke.json")
    cfg.cfm.dynamics_order = 1
    model = SecondOrderCFM(cfg.cfm)
    condition = make_condition(cfg, batch=1)
    condition.previous_velocity.normal_()
    state = model(condition)
    assert state.position.shape == condition.previous_points.shape
    assert torch.isfinite(state.velocity).all()
    assert torch.count_nonzero(state.acceleration) == 0
    target = torch.rand(1, cfg.cfm.num_points, 3)
    loss = compute_cfm_loss(model, condition, target, cfg.cfm_loss, torch.Generator().manual_seed(3))
    assert torch.isfinite(loss.total)
