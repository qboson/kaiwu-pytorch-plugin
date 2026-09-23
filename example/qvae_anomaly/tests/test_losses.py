"""Test loss functions."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from model import Config, build_model


def test_loss_terms_keys():
    """Test loss_terms returns all required keys."""
    cfg = Config(input_dimension=30, hidden_dim=64, num_var1=32, num_var2=32,
                 batch_size=32, epochs=1)
    model = build_model(cfg)
    x = torch.randn(16, 30)
    terms = model.loss_terms(x, kl_beta=0.1)
    required = ["total_loss", "recon_loss", "kl_loss", "cls_loss"]
    for k in required:
        assert k in terms, f"missing key: {k}"
    print("test_loss_terms_keys: OK")


def test_loss_scalar():
    """Test all losses are scalars."""
    cfg = Config(input_dimension=30, hidden_dim=64, num_var1=32, num_var2=32,
                 batch_size=32, epochs=1)
    model = build_model(cfg)
    x = torch.randn(16, 30)
    terms = model.loss_terms(x, kl_beta=0.1)
    assert terms["total_loss"].dim() == 0, "total_loss should be scalar"
    assert terms["recon_loss"].dim() == 0, "recon_loss should be scalar"
    print("test_loss_scalar: OK")


def test_loss_grad():
    """Test loss has gradient."""
    cfg = Config(input_dimension=30, hidden_dim=64, num_var1=32, num_var2=32,
                 batch_size=32, epochs=1)
    model = build_model(cfg)
    x = torch.randn(16, 30)
    terms = model.loss_terms(x, kl_beta=0.1)
    terms["total_loss"].backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "no gradient"
    print("test_loss_grad: OK")


if __name__ == "__main__":
    test_loss_terms_keys()
    test_loss_scalar()
    test_loss_grad()
    print("\nAll loss tests passed!")
