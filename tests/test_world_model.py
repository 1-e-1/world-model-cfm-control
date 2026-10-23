import torch

from codex_zbot.config import load_config
from codex_zbot.world_model.model import BeliefWorldModel, ReconstructionEncoder, world_model_loss


def test_reconstruction_encoder_and_world_rollout():
    cfg = load_config("configs/smoke.json")
    batch, points = 3, 32
    encoder = ReconstructionEncoder(cfg.world_model.latent_dim, point_dim=32)
    position = torch.rand(batch, points, 3)
    velocity = torch.rand(batch, points, 3) * 0.01
    confidence = torch.rand(batch, points, 1)
    particle_type = torch.randint(0, 2, (batch, points, 1)).float()
    latent = encoder(position, velocity, confidence, particle_type)
    assert latent.shape == (batch, cfg.world_model.latent_dim)

    model = BeliefWorldModel(cfg.world_model)
    horizon = 3
    robot = torch.rand(batch, horizon, cfg.world_model.robot_state_dim)
    action = torch.rand(batch, horizon, cfg.world_model.action_dim)
    predictions = model.rollout(latent, robot, action)
    target_latents = torch.rand(batch, horizon, cfg.world_model.latent_dim)
    scalar = torch.rand(batch, horizon)
    total, terms = world_model_loss(predictions, target_latents, scalar, scalar, scalar > 0.5)
    assert torch.isfinite(total)
    assert set(terms) == {"latent", "coverage", "reconstruction", "uncertainty", "collision"}

