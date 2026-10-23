from pathlib import Path

from codex_zbot.config import load_config


def test_smoke_config_override():
    path = Path(__file__).parents[1] / "configs" / "smoke.json"
    cfg = load_config(path)
    assert cfg.cfm.num_points == 64
    assert cfg.world_model.latent_dim == 32
    assert cfg.ppo.num_envs == 2


def test_no_world_model_ablation_is_a_real_feature_removal():
    path = Path(__file__).parents[1] / "configs" / "ablation_no_world_model.json"
    cfg = load_config(path)
    assert cfg.world_model.enabled is False
    assert cfg.world_model.latent_dim == 128
