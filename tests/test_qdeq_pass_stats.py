"""Solver diagnostics should describe one consistent forward/backward pass."""

import torch
from torch import nn

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class LinearMap(nn.Module):
    """Contractive map with a known implicit gradient."""

    def forward(self, state, inputs):
        return 0.25 * state + inputs


def test_backward_of_older_graph_cannot_replace_latest_forward_diagnostics():
    model = QDEQ(LinearMap(), object(), QDEQConfig(f_thres=0, stop_mode="abs"))
    first_input = torch.tensor([[1.0]], requires_grad=True)
    second_input = torch.tensor([[2.0]], requires_grad=True)
    first = model(first_input)
    second = model(second_input)
    latest_forward = model.get_last_stats()
    assert latest_forward["forward_residual"] == 2.0

    first.sum().backward()

    torch.testing.assert_close(first_input.grad, torch.full_like(first_input, 4.0 / 3), atol=1e-3, rtol=0)
    assert model.get_last_stats() == latest_forward

    second.sum().backward()

    stats = model.get_last_stats()
    assert stats["forward_residual"] == 2.0
    assert stats["backward_residual"] < 1e-3
