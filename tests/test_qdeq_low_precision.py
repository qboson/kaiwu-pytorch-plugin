"""Regression tests for low-precision QDEQ implicit gradients."""

import pytest
import torch

from kaiwu.torch_plugin import QDEQ


class ScalarContraction(torch.nn.Module):
    """A contraction with an analytically known implicit gradient."""

    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=dtype))

    def forward(self, state, inputs):
        assert state.dtype == self.weight.dtype
        return self.weight * state + inputs


class UnusedSampler:
    """An exact starting point should need no Ising sampling."""

    def solve(self, matrix):
        raise AssertionError("The supplied initial state is already a fixed point")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_implicit_backward_preserves_dtype_and_matches_analytic_gradient(dtype):
    """Actual QDEQ backward must work without changing layer or gradient dtype."""
    layer = ScalarContraction(dtype)
    model = QDEQ(layer, UnusedSampler())
    inputs = torch.tensor([[0.75, 0.375]], dtype=dtype, requires_grad=True)
    initial = inputs.detach() / 0.75

    output = model(inputs, z0=initial)
    output.sum().backward()

    assert output.dtype == inputs.grad.dtype == layer.weight.grad.dtype == dtype
    torch.testing.assert_close(output, initial, rtol=0, atol=0)
    torch.testing.assert_close(
        inputs.grad.float(), torch.full((1, 2), 4 / 3), rtol=0.02, atol=0.002
    )
    torch.testing.assert_close(
        layer.weight.grad.float(), torch.tensor(2.0), rtol=0.02, atol=0.002
    )
    assert model.get_last_stats()["backward_residual"] < 0.01
