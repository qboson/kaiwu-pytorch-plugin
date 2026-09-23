"""Test CleanEnergyQVAE model interface."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from model import Config, build_model


def test_build_model():
    """Test model builds with correct shapes."""
    cfg = Config(
        input_dimension=30,
        hidden_dim=64,
        num_var1=32,
        num_var2=32,
        batch_size=32,
        epochs=1,
    )
    model = build_model(cfg)
    assert model is not None
    assert cfg.input_dimension == 30
    print("test_build_model: OK")


def test_forward():
    """Test forward pass."""
    cfg = Config(
        input_dimension=30,
        hidden_dim=64,
        num_var1=32,
        num_var2=32,
        batch_size=32,
        epochs=1,
    )
    model = build_model(cfg)
    x = torch.randn(16, 30)
    terms = model.loss_terms(x, kl_beta=0.1)
    assert "total_loss" in terms
    assert "recon_loss" in terms
    print("test_forward: OK")


def test_score_energy():
    """Test energy score shape."""
    cfg = Config(
        input_dimension=30,
        hidden_dim=64,
        num_var1=32,
        num_var2=32,
        batch_size=32,
        epochs=1,
    )
    model = build_model(cfg)
    model.eval()
    x = torch.randn(16, 30)
    with torch.no_grad():
        energy = model.get_score_energy(x)
    assert energy.shape == (16,)
    print("test_score_energy: OK")


if __name__ == "__main__":
    test_build_model()
    test_forward()
    test_score_energy()
    print("\nAll model tests passed!")
