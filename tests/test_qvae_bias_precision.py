"""Clipped Bernoulli reconstruction biases stay finite after low-precision casts."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from kaiwu.torch_plugin import QVAE, RestrictedBoltzmannMachine


def make_model(dtype):
    """Build real QVAE components without constructing an external sampler."""
    config = SimpleNamespace(
        num_latent_units=2, loss_type="bernoulli", dist_beta=1.0,
        kl_beta=1.0, weight_decay=0.0,
    )
    model = QVAE(
        3, None, config, encoder=torch.nn.Linear(3, 2),
        decoder=torch.nn.Linear(2, 3),
        bm=RestrictedBoltzmannMachine(1, 1, device="cpu"), sampler=object(),
    ).to(dtype=dtype)
    return model


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_clipped_bias_is_finite_and_matches_probability_oracle(dtype):
    """Constant on/off features retain the specified 0.001/0.999 clipping policy."""
    with torch.random.fork_rng():
        model = make_model(dtype)
    means = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
    before = means.detach().clone()
    model.set_train_bias(means)
    bias = model.state_dict()["_train_bias"]
    expected = torch.logit(means.detach().clamp(0.001, 0.999)).to(dtype)
    assert bias.dtype == dtype
    assert torch.isfinite(bias).all()
    torch.testing.assert_close(bias, expected)
    torch.testing.assert_close(means.detach(), before)
    assert not bias.requires_grad

    # Exercise the bias's actual reconstruction-logit role without invoking the
    # unrelated mixed-precision posterior or any external negative-phase solver.
    decoded = model.decoder(torch.zeros(2, 2, dtype=dtype))
    loss = functional.binary_cross_entropy_with_logits(
        decoded + bias, torch.tensor([[0., 0., 1.], [0., 1., 1.]], dtype=dtype),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.decoder.weight.grad).all()
    assert torch.isfinite(model.decoder.bias.grad).all()


def test_bfloat16_scalar_bias_expands_without_infinity():
    """A scalar all-one mean is clipped before bfloat16 rounding."""
    with torch.random.fork_rng():
        model = make_model(torch.bfloat16)
    model.set_train_bias(1.0)
    bias = model.state_dict()["_train_bias"]
    torch.testing.assert_close(bias, torch.full((3,), 6.90625, dtype=torch.bfloat16))
