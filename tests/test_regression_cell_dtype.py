"""Batch conditions must not promote a converted cell decoder's inputs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn


def build_model(dtype, n_batches=2):
    path = Path(__file__).resolve().parents[1] / "example/qvae_cell/models.py"
    spec = importlib.util.spec_from_file_location("cell_condition_dtype_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = SimpleNamespace(num_latent_units=3, loss_type="mse", dist_beta=10.,
                          kl_beta=0., weight_decay=0.)
    return module.CellQVAE(4, nn.ReLU(), cfg, encoder=nn.Linear(4, 3),
                           decoder=nn.Linear(3 + n_batches, 4), bm=nn.Identity(),
                           sampler=object(), n_batches=n_batches).to(dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_conditioned_decoder_preserves_dtype_and_backward(dtype):
    model = build_model(dtype)
    zeta = torch.randn(2, 3, dtype=dtype, requires_grad=True)
    actual = model._decoder_input(zeta, torch.tensor([1, 0]))
    expected = torch.cat([zeta, zeta.new_tensor([[0, 1], [1, 0]])], dim=-1)
    torch.testing.assert_close(actual, expected)
    output = model.decoder(actual)
    output.float().square().sum().backward()
    assert torch.isfinite(zeta.grad).all()
    assert torch.count_nonzero(zeta.grad) > 0


def test_unconditioned_input_is_unchanged():
    model = build_model(torch.float64, n_batches=0)
    zeta = torch.randn(2, 3, dtype=torch.float64)
    assert model._decoder_input(zeta) is zeta


def test_missing_batch_indices_remain_an_error():
    with pytest.raises(ValueError, match="batch_idx"):
        build_model(torch.float32)._decoder_input(torch.randn(2, 3))
