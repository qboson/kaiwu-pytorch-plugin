"""The Anderson solver must return its best iterate with matching statistics."""

import pytest
import torch

from kaiwu.torch_plugin import QDEQConfig
from kaiwu.torch_plugin.qdeq import _anderson_solver


def test_solution_and_statistics_describe_the_same_iterate():
    bias = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=torch.float64)

    def func(value):
        return 0.5 * value + bias

    solution, stats = _anderson_solver(
        func, torch.zeros_like(bias), QDEQConfig(), threshold=4, eps=0.0
    )

    torch.testing.assert_close(solution, 2.0 * bias, rtol=1e-4, atol=1e-4)
    assert stats["nstep"] == 2
    residual = (func(solution) - solution).norm() / func(solution).norm()
    assert stats["residual"] == pytest.approx(residual.item(), rel=1e-6)
