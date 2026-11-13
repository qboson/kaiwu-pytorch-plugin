"""Regression tests for the documented QDEQ backward iteration budget.

Issue: ``QDEQConfig.b_thres`` is documented as the maximum number of backward
solver steps, but ``_anderson_solver`` stopped after ``b_thres - 2`` iterations
for every threshold of three or more. A caller asking for a tight but feasible
tolerance could therefore miss convergence even though the requested budget was
sufficient.
"""

import torch
from torch import nn

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class DiagonalLayer(nn.Module):
    """Contraction with known fixed point and analytic implicit gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.diag(torch.tensor([0.8, 0.7, 0.6, 0.5], dtype=torch.float64))
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return z @ self.weight.T + x


class NoSampler:
    """The exact initial fixed point means the forward solver is never used."""

    def solve(self, matrix):
        raise AssertionError("the exact initial state needs no sampler")


def run_backward(b_thres: int, b_eps: float) -> dict[str, float]:
    """Runs one exact-fixed-point forward pass and its implicit backward pass."""
    size = 4
    weight = torch.diag(torch.tensor([0.8, 0.7, 0.6, 0.5], dtype=torch.float64))
    x = torch.randn(1, size, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    exact = torch.linalg.solve(torch.eye(size, dtype=torch.float64) - weight, x.T).T

    model = QDEQ(
        DiagonalLayer(),
        NoSampler(),
        QDEQConfig(f_thres=40, f_eps=1e-14, b_thres=b_thres, b_eps=b_eps),
    )
    inputs = x.clone().requires_grad_()
    output = model(inputs, z0=exact.clone())
    output.backward(torch.ones_like(output))
    return model.get_last_stats()


def test_backward_spends_the_documented_iteration_budget() -> None:
    """A non-convergent solve must use exactly ``b_thres`` iterations."""
    for b_thres in (8, 10):
        stats = run_backward(b_thres=b_thres, b_eps=1e-300)
        assert stats["backward_nstep"] == b_thres


def test_tight_feasible_budget_reaches_tolerance() -> None:
    """Ten steps reach 3.5e-4; the truncated eight-step budget cannot."""
    stats = run_backward(b_thres=10, b_eps=3.5e-4)
    assert stats["backward_nstep"] == 10
    assert stats["backward_residual"] < 3.5e-4
