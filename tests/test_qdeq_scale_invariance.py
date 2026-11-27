"""Scale-equivariance and gradient-homogeneity regressions for QDEQ."""

import math

import pytest
import torch
from torch import nn

import kaiwu as kw
from kaiwu.torch_plugin import QDEQ, QDEQConfig


class RotationLinearLayer(nn.Module):
    """Contractive 2x2 rotation-scaled linear fixed-point map ``W z + x``."""

    def __init__(
        self,
        radius: float = 0.85,
        angle: float = 0.7,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        matrix = radius * torch.tensor(
            [[cos_a, -sin_a], [sin_a, cos_a]], dtype=dtype
        )
        self.weight = nn.Parameter(matrix)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluates ``z @ W.T + x``."""
        return z @ self.weight.T + x


class ZeroMapLayer(nn.Module):
    """Contractive constant-zero fixed-point map ``f(z, x) = 0``."""

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Returns zeros shaped like ``z``."""
        del x
        return torch.zeros_like(z)


@pytest.mark.parametrize("scale", [1.0, 1e-12, 1e6])
def test_forward_linear_equivariance_across_input_scales(scale: float) -> None:
    """Linear fixed points satisfy z*(c * x) == c * (I - W)^-1 x across scales."""
    layer = RotationLinearLayer(radius=0.85, dtype=torch.float64)
    config = QDEQConfig(
        num_basis=2,
        coeff_bits=6,
        f_thres=25,
        f_eps=1e-4,
        stop_mode="rel",
    )
    deq = QDEQ(layer, kw.classical.BruteForceOptimizer(), config).eval()
    base_x = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    exact_base = torch.linalg.solve(eye - layer.weight.detach(), base_x.T).T

    torch.manual_seed(0)
    actual = deq(scale * base_x)
    stats = deq.get_last_stats()

    torch.testing.assert_close(actual, scale * exact_base, rtol=5e-4, atol=0.0)
    assert stats["forward_residual"] < 1e-4


def test_forward_float32_large_input_does_not_underflow_finite_difference() -> None:
    """Finite-difference Jacobian probing must not vanish when ||z|| >> fd_eps in FP32."""
    layer = RotationLinearLayer(radius=0.5, dtype=torch.float32)
    config = QDEQConfig(
        num_basis=2,
        coeff_bits=6,
        f_thres=20,
        f_eps=1e-3,
        stop_mode="rel",
    )
    deq = QDEQ(layer, kw.classical.BruteForceOptimizer(), config).eval()
    x = torch.tensor([[1e8, -5e7]], dtype=torch.float32)
    eye = torch.eye(2, dtype=torch.float32)
    exact = torch.linalg.solve(eye - layer.weight.detach(), x.T).T

    torch.manual_seed(0)
    actual = deq(x, z0=0.5 * x)
    rel_error = ((actual - exact).norm() / exact.norm()).item()

    assert rel_error < 2e-3
    assert deq.get_last_stats()["forward_residual"] < 1e-3


@pytest.mark.parametrize("loss_scale", [1.0, 1e-4, 1e-8, 1e4])
def test_backward_implicit_gradient_homogeneity_across_loss_scales(
    loss_scale: float,
) -> None:
    """Scaling the scalar loss by c must scale implicit x and weight gradients by c."""
    layer = RotationLinearLayer(radius=0.9, dtype=torch.float64)
    config = QDEQConfig(
        num_basis=2,
        coeff_bits=6,
        f_thres=25,
        f_eps=1e-9,
        b_thres=25,
        b_eps=1e-8,
    )
    deq = QDEQ(layer, kw.classical.BruteForceOptimizer(), config).train()
    x = torch.tensor([[1.0, -0.5]], dtype=torch.float64, requires_grad=True)
    grad_weights = torch.tensor([[1.0, -2.0]], dtype=torch.float64)

    eye = torch.eye(2, dtype=torch.float64)
    z_exact = torch.linalg.solve(eye - layer.weight.detach(), x.detach().T).T
    out = deq(x, z0=z_exact)
    (loss_scale * (out * grad_weights).sum()).backward()

    upstream = loss_scale * grad_weights
    expected_x_grad = torch.linalg.solve(
        eye - layer.weight.detach().T, upstream.T
    ).T
    expected_w_grad = expected_x_grad.T @ z_exact

    torch.testing.assert_close(x.grad, expected_x_grad, rtol=1e-4, atol=0.0)
    torch.testing.assert_close(
        layer.weight.grad, expected_w_grad, rtol=1e-4, atol=0.0
    )


def test_backward_mixed_scale_and_zero_gradient_batch_rows() -> None:
    """Per-sample Anderson damping handles O(1), O(1e-6), and exact-zero rows together."""
    layer = RotationLinearLayer(radius=0.9, dtype=torch.float64)
    config = QDEQConfig(
        num_basis=2,
        coeff_bits=6,
        f_thres=25,
        f_eps=1e-9,
        b_thres=25,
        b_eps=1e-8,
    )
    deq = QDEQ(layer, kw.classical.BruteForceOptimizer(), config).train()
    x = torch.tensor(
        [[1.0, -0.5], [-0.25, 0.75], [0.5, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    row_weights = torch.tensor(
        [[1.0, -0.5], [1e-6, 2e-6], [0.0, 0.0]],
        dtype=torch.float64,
    )

    eye = torch.eye(2, dtype=torch.float64)
    z_exact = torch.linalg.solve(eye - layer.weight.detach(), x.detach().T).T
    out = deq(x, z0=z_exact)
    (out * row_weights).sum().backward()

    expected_x_grad = torch.linalg.solve(
        eye - layer.weight.detach().T, row_weights.T
    ).T
    torch.testing.assert_close(x.grad[0], expected_x_grad[0], rtol=1e-4, atol=0.0)
    torch.testing.assert_close(x.grad[1], expected_x_grad[1], rtol=1e-4, atol=0.0)
    torch.testing.assert_close(x.grad[2], torch.zeros(2, dtype=torch.float64))
    assert math.isfinite(deq.get_last_stats()["backward_residual"])


def test_zero_fixed_point_and_zero_map_residuals_remain_finite() -> None:
    """Exact-zero fixed points and zero-valued maps produce finite residuals without NaN."""
    layer = RotationLinearLayer(radius=0.5, dtype=torch.float64)
    deq = QDEQ(layer, kw.classical.BruteForceOptimizer()).eval()
    zeros = torch.zeros(2, 2, dtype=torch.float64)
    out_zero = deq(zeros)
    stats_zero = deq.get_last_stats()

    torch.testing.assert_close(out_zero, zeros)
    assert stats_zero["forward_nstep"] == 0
    assert stats_zero["forward_residual"] == 0.0

    zero_deq = QDEQ(
        ZeroMapLayer(),
        kw.classical.BruteForceOptimizer(),
        QDEQConfig(num_basis=2, coeff_bits=4, f_thres=10, f_eps=1e-4),
    ).eval()
    z0_nonzero = torch.tensor([[2.0, -3.0]], dtype=torch.float64)
    torch.manual_seed(0)
    out_from_nonzero = zero_deq(torch.zeros_like(z0_nonzero), z0=z0_nonzero)
    stats_zero_map = zero_deq.get_last_stats()

    torch.testing.assert_close(out_from_nonzero, torch.zeros_like(z0_nonzero))
    assert math.isfinite(stats_zero_map["forward_residual"])
