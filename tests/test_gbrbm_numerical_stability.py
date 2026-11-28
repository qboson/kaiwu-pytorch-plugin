"""Numerical-stability and low-precision regressions for GBRBM."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import kaiwu as kw
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def _make_reference_fp64(
    mu: list[float],
    log_var: list[float],
    quadratic_coef: list[list[float]],
    linear_bias: list[float],
    eps: float = 1e-8,
) -> GaussianBernoulliRestrictedBoltzmannMachine:
    """Construct a float64 reference GBRBM with identical numerical parameters."""
    num_gaussian = len(mu)
    num_bernoulli = len(linear_bias)
    ref = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=num_gaussian,
        num_hidden=num_bernoulli,
        eps=eps,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        ref.mu.copy_(torch.tensor(mu, dtype=torch.float64))
        ref.log_var.copy_(torch.tensor(log_var, dtype=torch.float64))
        ref.quadratic_coef.copy_(torch.tensor(quadratic_coef, dtype=torch.float64))
        ref.linear_bias.copy_(torch.tensor(linear_bias, dtype=torch.float64))
    return ref


def test_float32_large_offset_quadratic_energy_and_gradients_stay_finite() -> None:
    """Whitened (v - mu) / std avoids premature (v - mu)^2 overflow in float32."""
    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=1,
        num_hidden=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bm.mu.copy_(torch.tensor([-2e19], dtype=torch.float32))
        bm.log_var.copy_(torch.tensor([math.log(1e38)], dtype=torch.float32))
        bm.quadratic_coef.zero_()
        bm.linear_bias.zero_()

    s_gaussian = torch.tensor([[2e19]], dtype=torch.float32, requires_grad=True)
    s_all = torch.cat([s_gaussian, torch.zeros((1, 1), dtype=torch.float32)], dim=-1)

    energy = bm(s_all)
    energy.backward()

    torch.testing.assert_close(
        energy, torch.tensor([8.0], dtype=torch.float32), rtol=1e-5, atol=1e-4
    )
    torch.testing.assert_close(
        bm.mu.grad, torch.tensor([-4e-19], dtype=torch.float32), rtol=1e-5, atol=0.0
    )
    torch.testing.assert_close(
        bm.log_var.grad, torch.tensor([-8.0], dtype=torch.float32), rtol=1e-5, atol=1e-4
    )
    torch.testing.assert_close(
        s_gaussian.grad, torch.tensor([[4e-19]], dtype=torch.float32), rtol=1e-5, atol=0.0
    )
    torch.testing.assert_close(
        bm.marginal_energy(s_gaussian.detach()),
        torch.tensor([8.0 - math.log(2.0)], dtype=torch.float32),
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(
        bm.positive_phase_energy_expectation(s_gaussian.detach()),
        torch.tensor([8.0], dtype=torch.float32),
        rtol=1e-5,
        atol=1e-4,
    )


def test_float32_diff_overflow_preserves_exact_centered_gradients() -> None:
    """Even when s_gaussian - mu overflows float32, forward and all grads stay exact."""
    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=1,
        num_hidden=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bm.mu.copy_(torch.tensor([-2e38], dtype=torch.float32))
        bm.log_var.copy_(torch.tensor([175.0], dtype=torch.float32))
        bm.quadratic_coef.zero_()
        bm.linear_bias.zero_()

    s_gaussian = torch.tensor([[2e38]], dtype=torch.float32, requires_grad=True)
    s_all = torch.cat([s_gaussian, torch.zeros((1, 1), dtype=torch.float32)], dim=-1)

    energy = bm(s_all)
    energy.backward()

    centered_ref = (float(s_gaussian.item()) - float(bm.mu.item())) * math.exp(-87.5)
    energy_ref = 0.5 * centered_ref * centered_ref
    state_grad_ref = centered_ref * math.exp(-87.5)

    torch.testing.assert_close(
        energy, torch.tensor([energy_ref], dtype=torch.float32), rtol=1e-5, atol=1e-4
    )
    torch.testing.assert_close(
        bm.log_var.grad,
        torch.tensor([-energy_ref], dtype=torch.float32),
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(
        s_gaussian.grad,
        torch.tensor([[state_grad_ref]], dtype=torch.float32),
        rtol=1e-5,
        atol=0.0,
    )
    torch.testing.assert_close(
        bm.mu.grad,
        torch.tensor([-state_grad_ref], dtype=torch.float32),
        rtol=1e-5,
        atol=0.0,
    )

    bm.zero_grad()
    s_gaussian.grad = None
    pos = bm.positive_phase_energy_expectation(s_gaussian)
    pos.backward()
    torch.testing.assert_close(
        pos, torch.tensor([energy_ref], dtype=torch.float32), rtol=1e-5, atol=1e-4
    )
    torch.testing.assert_close(
        bm.log_var.grad,
        torch.tensor([-energy_ref], dtype=torch.float32),
        rtol=1e-5,
        atol=1e-4,
    )
    assert torch.isfinite(s_gaussian.grad).all()
    assert torch.isfinite(bm.mu.grad).all()


def test_float32_small_variance_whitened_contraction_avoids_premature_overflow() -> None:
    """Balanced precision scaling avoids s_gaussian / var overflow across methods."""
    mu = [1e31]
    log_var = [-18.0]  # var = 1.523e-8 > eps=1e-8 (strictly inside in-bounds range)
    quadratic = [[1e-35, -1e-35]]
    bias = [0.25, -0.5]

    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=1,
        num_hidden=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bm.mu.copy_(torch.tensor(mu, dtype=torch.float32))
        bm.log_var.copy_(torch.tensor(log_var, dtype=torch.float32))
        bm.quadratic_coef.copy_(torch.tensor(quadratic, dtype=torch.float32))
        bm.linear_bias.copy_(torch.tensor(bias, dtype=torch.float32))

    ref = _make_reference_fp64(
        [float(v) for v in bm.mu.detach().double()],
        [float(v) for v in bm.log_var.detach().double()],
        [[float(v) for v in row] for row in bm.quadratic_coef.detach().double()],
        [float(v) for v in bm.linear_bias.detach().double()],
    )
    s_g32 = bm.mu.detach().clone().unsqueeze(0).requires_grad_(True)
    s_g64 = s_g32.detach().to(torch.float64).requires_grad_(True)
    s_all32 = torch.cat([s_g32, torch.zeros((1, 2), dtype=torch.float32)], dim=-1)
    s_all64 = s_all32.detach().to(torch.float64)

    e32 = bm(s_all32)
    torch.testing.assert_close(
        e32, ref(s_all64).to(torch.float32), rtol=1e-5, atol=1e-5
    )
    e32.sum().backward()
    assert torch.isfinite(s_g32.grad).all()
    for param in bm.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()

    bm.zero_grad()
    s_g32.grad = None
    torch.testing.assert_close(
        bm.marginal_energy(s_g32.detach()),
        ref.marginal_energy(s_g64.detach()).to(torch.float32),
        rtol=1e-5,
        atol=1e-2,
    )

    pos32 = bm.positive_phase_energy_expectation(s_g32)
    pos64 = ref.positive_phase_energy_expectation(s_g64)
    torch.testing.assert_close(pos32, pos64.to(torch.float32), rtol=1e-5, atol=1e-2)
    pos32.sum().backward()
    assert torch.isfinite(s_g32.grad).all()
    assert torch.isfinite(bm.mu.grad).all()
    assert torch.isfinite(bm.log_var.grad).all()
    assert torch.isfinite(bm.linear_bias.grad).all()
    assert not torch.isnan(bm.quadratic_coef.grad).any()

    ising32 = bm.get_ising_matrix()
    ising64 = ref.get_ising_matrix()
    assert np.isfinite(ising32).all()
    np.testing.assert_allclose(ising32, ising64, rtol=1e-5, atol=1e-3)


def test_zero_gaussian_and_all_inactive_bernoulli_avoid_nan_in_forward_and_backward() -> None:
    """Zero Gaussian inputs and all-inactive Bernoulli masks never trigger 0 * inf = NaN."""
    bm_zero_g = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=1,
        num_hidden=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bm_zero_g.mu.zero_()
        bm_zero_g.log_var.copy_(torch.tensor([-18.0], dtype=torch.float32))
        bm_zero_g.quadratic_coef.copy_(
            torch.tensor([[1e-4, -1e36]], dtype=torch.float32)
        )
        bm_zero_g.linear_bias.zero_()

    s_g0 = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
    s_all0 = torch.cat([s_g0, torch.ones((1, 2), dtype=torch.float32)], dim=-1)
    e0 = bm_zero_g(s_all0)
    e0.backward()

    assert e0.item() == 0.0
    assert bm_zero_g.log_var.grad.item() == 0.0
    assert torch.all(bm_zero_g.quadratic_coef.grad == 0.0)
    assert not torch.isnan(s_g0.grad).any()

    bm_zero_g.zero_grad()
    s_g0.grad = None
    pos0 = bm_zero_g.positive_phase_energy_expectation(s_g0)
    pos0.backward()
    assert pos0.item() == 0.0
    assert bm_zero_g.log_var.grad.item() == 0.0
    assert torch.all(bm_zero_g.quadratic_coef.grad == 0.0)
    assert not torch.isnan(s_g0.grad).any()
    torch.testing.assert_close(
        bm_zero_g.marginal_energy(s_g0.detach()),
        torch.tensor([-2.0 * math.log(2.0)], dtype=torch.float32),
    )
    torch.testing.assert_close(
        bm_zero_g.infer_from_gaussian(s_g0.detach(), binarize=False),
        torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float32),
    )

    # All-inactive Bernoulli / all-saturated negative logits with huge W and s_gaussian:
    bm_inact = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=1,
        num_hidden=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bm_inact.mu.copy_(torch.tensor([1e31], dtype=torch.float32))
        bm_inact.log_var.copy_(torch.tensor([-18.0], dtype=torch.float32))
        bm_inact.quadratic_coef.copy_(
            torch.tensor([[-1e35, -1e35]], dtype=torch.float32)
        )
        bm_inact.linear_bias.zero_()

    s_gi = torch.tensor([[1e31]], dtype=torch.float32, requires_grad=True)
    s_alli = torch.cat([s_gi, torch.zeros((1, 2), dtype=torch.float32)], dim=-1)
    ei = bm_inact(s_alli)
    ei.backward()
    assert ei.item() == 0.0
    assert torch.isfinite(s_gi.grad).all()
    for param in bm_inact.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()

    bm_inact.zero_grad()
    s_gi.grad = None
    posi = bm_inact.positive_phase_energy_expectation(s_gi)
    posi.backward()
    assert posi.item() == 0.0
    assert torch.isfinite(s_gi.grad).all()
    for param in bm_inact.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inactive_bernoulli_and_saturated_negative_logits_do_not_produce_nan(
    dtype: torch.dtype,
) -> None:
    """Inactive h_j=0 and saturated sigmoid(-inf)=0 must contribute 0 instead of NaN."""
    scale_v = 1e25 if dtype == torch.float32 else 1e200
    small_w = 1e-24 if dtype == torch.float32 else 1e-199
    huge_neg_w = -1e20 if dtype == torch.float32 else -1e200

    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=1,
        num_hidden=2,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bm.mu.copy_(torch.tensor([scale_v], dtype=dtype))
        bm.log_var.zero_()  # unit variance var == 1.0
        bm.quadratic_coef.copy_(torch.tensor([[small_w, huge_neg_w]], dtype=dtype))
        bm.linear_bias.zero_()

    s_all = torch.tensor([[scale_v, 1.0, 0.0]], dtype=dtype, requires_grad=True)
    energy = bm(s_all)
    energy.backward()

    torch.testing.assert_close(
        energy, torch.tensor([-10.0], dtype=dtype), rtol=1e-5, atol=1e-4
    )
    assert torch.isfinite(s_all.grad).all()
    assert torch.isfinite(bm.mu.grad).all()
    assert torch.isfinite(bm.log_var.grad).all()
    assert bm.quadratic_coef.grad[0, 1].item() == 0.0

    bm.zero_grad()
    s_gaussian = torch.tensor([[scale_v]], dtype=dtype, requires_grad=True)
    expected_pos = -10.0 / (1.0 + math.exp(-10.0))
    pos_energy = bm.positive_phase_energy_expectation(s_gaussian)
    pos_energy.backward()

    torch.testing.assert_close(
        pos_energy,
        torch.tensor([expected_pos], dtype=dtype),
        rtol=1e-5,
        atol=1e-4,
    )
    assert torch.isfinite(s_gaussian.grad).all()
    assert torch.isfinite(bm.mu.grad).all()
    assert torch.isfinite(bm.log_var.grad).all()
    assert bm.quadratic_coef.grad[0, 1].item() == 0.0


@pytest.mark.parametrize("eps", [1e-4, 1e-8])
def test_float16_preserves_finite_energies_gradients_ising_and_sampling(
    eps: float,
) -> None:
    """FP16 GBRBMs avoid intermediate v/var and 1/var overflow across all methods."""
    dtype = torch.float16
    mu = [8.0, -3.0]
    log_var = [math.log(1e-4), math.log(4e-4)]
    quadratic = [[-2.5e-5, -1e-3], [5e-4, 1e-3]]
    bias = [0.25, -0.5]

    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=2,
        num_hidden=2,
        eps=eps,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bm.mu.copy_(torch.tensor(mu, dtype=dtype))
        bm.log_var.copy_(torch.tensor(log_var, dtype=dtype))
        bm.quadratic_coef.copy_(torch.tensor(quadratic, dtype=dtype))
        bm.linear_bias.copy_(torch.tensor(bias, dtype=dtype))

    ref = _make_reference_fp64(
        [float(v) for v in bm.mu.detach().double()],
        [float(v) for v in bm.log_var.detach().double()],
        [[float(v) for v in row] for row in bm.quadratic_coef.detach().double()],
        [float(v) for v in bm.linear_bias.detach().double()],
        eps=eps,
    )

    s_g = torch.tensor([[8.02, -2.99]], dtype=dtype, requires_grad=True)
    s_g_ref = s_g.detach().double()
    s_all = torch.cat([s_g, torch.tensor([[1.0, 0.0]], dtype=dtype)], dim=-1)
    s_all_ref = s_all.detach().double()

    energy = bm(s_all)
    assert energy.dtype == dtype
    assert torch.isfinite(energy).all()
    torch.testing.assert_close(
        energy.double(), ref(s_all_ref), rtol=2e-2, atol=5e-2
    )

    marginal = bm.marginal_energy(s_g.detach())
    assert marginal.dtype == dtype
    assert torch.isfinite(marginal).all()
    torch.testing.assert_close(
        marginal.double(), ref.marginal_energy(s_g_ref), rtol=2e-2, atol=5e-2
    )

    inferred = bm.infer_from_gaussian(s_g.detach(), binarize=False)
    assert inferred.dtype == dtype
    assert torch.isfinite(inferred).all()
    torch.testing.assert_close(
        inferred[:, 2:].double(),
        ref.infer_from_gaussian(s_g_ref, binarize=False)[:, 2:],
        rtol=2e-2,
        atol=2e-2,
    )

    pos = bm.positive_phase_energy_expectation(s_g)
    assert pos.dtype == dtype
    assert torch.isfinite(pos).all()
    torch.testing.assert_close(
        pos.double(),
        ref.positive_phase_energy_expectation(s_g_ref),
        rtol=2e-2,
        atol=5e-2,
    )
    pos.sum().backward()
    assert torch.isfinite(s_g.grad).all()
    for param in bm.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()

    ising_mat = bm.get_ising_matrix()
    assert ising_mat.dtype == np.float16
    assert np.isfinite(ising_mat).all()
    np.testing.assert_allclose(
        ising_mat.astype(np.float64), ref.get_ising_matrix(), rtol=2e-2, atol=2e-2
    )

    sampled = bm.sample(kw.classical.BruteForceOptimizer())
    assert sampled.dtype == dtype
    assert torch.isfinite(sampled).all()


def test_float16_ising_matrix_retains_float32_when_couplings_exceed_fp16_max() -> None:
    """FP16 GBRBM retains finite float32 Ising matrix when entries exceed 65,504."""
    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=1,
        num_hidden=2,
        eps=1e-4,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bm.mu.copy_(torch.tensor([10.0], dtype=torch.float16))
        bm.log_var.copy_(torch.tensor([math.log(1e-4)], dtype=torch.float16))
        bm.quadratic_coef.copy_(torch.tensor([[4.0, -4.0]], dtype=torch.float16))
        bm.linear_bias.zero_()

    ref = _make_reference_fp64(
        [float(bm.mu.item())],
        [float(bm.log_var.item())],
        [[float(v) for v in bm.quadratic_coef.detach().double()[0]]],
        [0.0, 0.0],
        eps=1e-4,
    )
    ising_mat = bm.get_ising_matrix()
    assert ising_mat.dtype == np.float32
    assert np.isfinite(ising_mat).all()
    assert float(np.abs(ising_mat).max()) > float(torch.finfo(torch.float16).max)
    np.testing.assert_allclose(ising_mat, ref.get_ising_matrix(), rtol=1e-4, atol=1.0)

    sampled = bm.sample(kw.classical.BruteForceOptimizer())
    assert sampled.dtype == torch.float16
    assert torch.isfinite(sampled).all()


def test_bfloat16_whitened_contractions_preserve_finite_energies_and_gradients() -> None:
    """BF16 tensor paths avoid premature v/var overflow and 0 * inf NaN poisoning."""
    dtype = torch.bfloat16
    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=1,
        num_hidden=2,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bm.mu.copy_(torch.tensor([1e31], dtype=dtype))
        bm.log_var.copy_(torch.tensor([-18.0], dtype=dtype))
        bm.quadratic_coef.copy_(torch.tensor([[-1e-34, -1e-34]], dtype=dtype))
        bm.linear_bias.zero_()

    ref = _make_reference_fp64(
        [float(bm.mu.item())],
        [float(bm.log_var.item())],
        [[float(v) for v in bm.quadratic_coef.detach().double()[0]]],
        [0.0, 0.0],
    )
    s_g = bm.mu.detach().clone().unsqueeze(0).requires_grad_(True)
    s_all = torch.cat([s_g, torch.zeros((1, 2), dtype=dtype)], dim=-1)

    energy = bm(s_all)
    assert energy.dtype == dtype
    torch.testing.assert_close(energy, torch.zeros(1, dtype=dtype))

    pos = bm.positive_phase_energy_expectation(s_g)
    assert pos.dtype == dtype
    assert torch.isfinite(pos).all()
    torch.testing.assert_close(
        pos.double(),
        ref.positive_phase_energy_expectation(s_g.detach().double()),
        rtol=2e-2,
        atol=5e-1,
    )
    pos.sum().backward()
    assert torch.isfinite(s_g.grad).all()
    for param in bm.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()


def test_variance_floor_clipping_preserves_zero_log_var_gradient() -> None:
    """Below log(eps), variance clamping continues to stop log_var gradients cleanly."""
    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=2,
        num_hidden=2,
        eps=1e-8,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bm.log_var.fill_(-30.0)
    s_gaussian = torch.tensor([[0.5, -0.25]], dtype=torch.float64)
    loss = bm.positive_phase_energy_expectation(s_gaussian).sum()
    loss.backward()

    assert torch.isfinite(loss)
    torch.testing.assert_close(
        bm.log_var.grad, torch.zeros_like(bm.log_var), rtol=0.0, atol=0.0
    )
