"""QDEQ matrix-free GMRES backward solver."""

import pytest
import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class LinearLayer(torch.nn.Module):
    """Layer ``z @ W.T + x`` with a known fixed point."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, z, x):
        return z @ self.weight.T + x


class UnusedSampler:
    """An exact initial state should not call the sampler."""

    def solve(self, matrix):
        raise AssertionError("the exact initial state needs no sampler")


def make_weight(size, top, seed):
    """Contraction with singular values spread over ``[0.2, top]``."""
    torch.manual_seed(seed)
    orthogonal, _ = torch.linalg.qr(torch.randn(size, size, dtype=torch.float64))
    singular = torch.linspace(top, 0.2, size, dtype=torch.float64)
    return orthogonal @ torch.diag(singular) @ orthogonal.T


def solve_config(**kwargs):
    """Tight tolerances so gradients can be compared with the analytic value."""
    defaults = {"f_thres": 40, "f_eps": 1e-10, "b_thres": 80, "b_eps": 1e-11}
    defaults.update(kwargs)
    return QDEQConfig(**defaults)


def run_backward(weight, seed, solver):
    """Returns the input gradient at the exact fixed point."""
    size = weight.shape[0]
    x = torch.randn(
        1,
        size,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )
    identity = torch.eye(size, dtype=torch.float64)
    fixed_point = torch.linalg.solve(identity - weight, x.T).T
    config = solve_config() if solver is None else solve_config(b_solver=solver)
    model = QDEQ(LinearLayer(weight), UnusedSampler(), config)
    inputs = x.clone().requires_grad_()
    output = model(inputs, z0=fixed_point.clone())
    upstream = torch.ones(1, size, dtype=torch.float64)
    output.backward(upstream)
    return inputs.grad, model, identity, upstream


@pytest.mark.parametrize("size,seed", [(4, 15), (8, 4), (16, 16)])
def test_gmres_backward_matches_analytic_gradient(size, seed):
    """GMRES reaches the analytic implicit gradient at tight tolerances."""
    weight = make_weight(size, 0.9, seed)
    gradient, model, identity, upstream = run_backward(weight, seed, "gmres")

    exact = torch.linalg.solve((identity - weight).T, upstream.T).T
    error = (gradient - exact).abs().max() / exact.abs().max()
    assert error < 1e-8
    assert model.get_last_stats()["backward_residual"] < 1e-8


def test_auto_falls_back_to_gmres_when_anderson_misses_the_tolerance():
    """The default solver must not return an unconverged gradient silently."""
    weight = make_weight(4, 0.9, 15)
    gradient, model, identity, upstream = run_backward(weight, 15, None)

    exact = torch.linalg.solve((identity - weight).T, upstream.T).T
    error = (gradient - exact).abs().max() / exact.abs().max()
    assert error < 1e-8
    assert model.get_last_stats()["backward_residual"] < 1e-8


def test_anderson_and_gmres_agree_on_a_well_conditioned_layer():
    """Both backward solvers return the same gradient when Anderson converges."""
    weight = make_weight(8, 0.5, 3)
    anderson_gradient, _, _, _ = run_backward(weight, 3, "anderson")
    gmres_gradient, _, _, _ = run_backward(weight, 3, "gmres")

    torch.testing.assert_close(anderson_gradient, gmres_gradient, rtol=1e-8, atol=1e-9)


def test_rejects_unknown_backward_solver():
    """Only the documented backward solver names are accepted."""
    with pytest.raises(ValueError, match="b_solver"):
        QDEQ(LinearLayer(torch.eye(2)), UnusedSampler(), QDEQConfig(b_solver="cg"))