"""Contract test for the QDEQ implicit-function backward hook."""

from __future__ import annotations

import torch

from kaiwu.torch_plugin.qdeq import QDEQ, QDEQConfig


class _FakeQDEQ:
    def __init__(self) -> None:
        self.config = QDEQConfig()
        self._last_stats: dict[str, float] = {}


def test_replaces_the_one_step_gradient_with_the_fixed_point_solution():
    module = _FakeQDEQ()
    scale = 0.5
    z_star = torch.tensor([0.0, 1.0], requires_grad=True)
    new_z = scale * z_star

    QDEQ._attach_implicit_backward(module, new_z, z_star)
    new_z.sum().backward()

    # The hook solves y = J^T y + g with J^T = scale and g = 1, so the gradient
    # reaching new_z is 1 / (1 - scale) = 2; back through new_z = scale * z_star
    # this becomes scale / (1 - scale) = 1.0 for every element.
    assert z_star.grad is not None
    expected = torch.full((2,), scale / (1.0 - scale))
    assert torch.allclose(z_star.grad, expected, atol=5e-3, rtol=5e-3)


def test_records_backward_solver_statistics():
    module = _FakeQDEQ()
    z_star = torch.tensor([0.0], requires_grad=True)
    new_z = 0.25 * z_star

    QDEQ._attach_implicit_backward(module, new_z, z_star)
    new_z.sum().backward()

    assert "backward_nstep" in module._last_stats
    assert "backward_residual" in module._last_stats
    assert all(key.startswith("backward_") for key in module._last_stats)
