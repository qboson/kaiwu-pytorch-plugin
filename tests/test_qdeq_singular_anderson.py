"""A singular Anderson system can fall back for one sample during backward."""

import torch
from torch import nn

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class LinearMap(nn.Module):
    def forward(self, latent, inputs):
        return 0.5 * latent + inputs


class ExactForwardQDEQ(QDEQ):
    def solve_fixed_point(self, func, z0):
        # For LinearMap, func(0) = inputs and z* = 2 * inputs.
        return 2 * func(torch.zeros_like(z0)), {}


def test_zero_gradient_row_does_not_break_other_rows_in_backward():
    model = ExactForwardQDEQ(
        LinearMap(), None, QDEQConfig(anderson_lam=0, b_thres=3, stop_mode="abs")
    )
    inputs = torch.tensor([[1.0], [3.0]], dtype=torch.float64, requires_grad=True)

    output = model(inputs)
    output[1].sum().backward()

    # Row 0 has identical zero residuals. Row 1 must still take an Anderson
    # step to reach 2 within this budget instead of a plain iteration to 1.5.
    torch.testing.assert_close(inputs.grad, torch.tensor([[0.0], [2.0]], dtype=inputs.dtype))
    assert model.get_last_stats()["backward_residual"] < 1e-12
