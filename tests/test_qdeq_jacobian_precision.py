# SPDX-License-Identifier: Apache-2.0
"""Check Jacobian penalties against the analytic scaled-identity Jacobian."""

import json
from unittest import mock

import pytest
import torch

from kaiwu.classical import BruteForceOptimizer
from kaiwu.torch_plugin import QDEQ, QDEQConfig


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(("vecs", "scale_value"), [(1, 1.0), (3, 0.5)])
def test_jacobian_penalty_and_gradient_match_scaled_identity(dtype, vecs, scale_value):
    """Large states and several probes must not overflow a finite mean penalty."""
    shape = (8, 128, 128)
    generator = torch.Generator().manual_seed(23)
    probes = [torch.randn(shape, dtype=dtype, generator=generator) for _ in range(vecs)]
    state = torch.zeros(shape, dtype=dtype, requires_grad=True)
    scale = torch.tensor(scale_value, dtype=dtype, requires_grad=True)

    # For J = scale * I, each probe contributes scale**2 * ||probe||**2 / N.
    probe_energy = sum(probe.double().square().sum().item() for probe in probes)
    probe_energy /= vecs * state.numel()
    expected_penalty = torch.tensor(scale_value**2 * probe_energy, dtype=torch.float64)
    expected_gradient = torch.tensor(2 * scale_value * probe_energy, dtype=torch.float64)

    with mock.patch("kaiwu.torch_plugin.qdeq.torch.randn_like", side_effect=probes):
        penalty = QDEQ.jacobian_loss(scale * state, state, vecs=vecs)

    assert torch.isfinite(penalty)
    # Large FP32 vector-norm reductions can differ by a few parts per million.
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(penalty.double(), expected_penalty, rtol=tolerance, atol=tolerance)
    expected_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    assert penalty.dtype == expected_dtype

    gradient = torch.autograd.grad(penalty, scale)[0]
    assert torch.isfinite(gradient)
    gradient_tolerance = {
        torch.float16: 2e-3,
        torch.bfloat16: 1e-2,
        torch.float32: 2e-6,
        torch.float64: 1e-12,
    }[dtype]
    torch.testing.assert_close(
        gradient.double(), expected_gradient,
        rtol=gradient_tolerance, atol=gradient_tolerance,
    )


class ScaledLayer(torch.nn.Module):
    """Contractive layer with an analytic zero-input fixed point and Jacobian."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float16))

    def forward(self, state, inputs):
        """Return ``scale * state + inputs``."""
        return self.scale * state + inputs


def test_forward_stores_finite_differentiable_half_precision_penalty():
    """The public forward path exposes a usable penalty without sampling at z*=0."""
    inputs = torch.zeros((8, 256, 256), dtype=torch.float16)
    probe = torch.randn(
        inputs.shape, dtype=inputs.dtype, generator=torch.Generator().manual_seed(7),
    )
    probe_energy = probe.double().square().mean().item()
    layer = ScaledLayer()
    sampler = mock.Mock()
    sampler.solve.side_effect = AssertionError("The zero fixed point needs no sampler call")
    model = QDEQ(layer, sampler, QDEQConfig(jac_loss_vecs=1))

    with mock.patch("kaiwu.torch_plugin.qdeq.torch.randn_like", return_value=probe):
        output = model(inputs)

    sampler.solve.assert_not_called()
    torch.testing.assert_close(output, inputs)
    penalty = model.last_jac_loss
    assert torch.isfinite(penalty)
    assert penalty.dtype == torch.float32
    torch.testing.assert_close(
        penalty.double(), torch.tensor(0.25 * probe_energy, dtype=torch.float64),
        rtol=2e-6, atol=2e-6,
    )
    penalty.backward()
    assert torch.isfinite(layer.scale.grad)
    torch.testing.assert_close(
        layer.scale.grad.double(), torch.tensor(probe_energy, dtype=torch.float64),
        rtol=2e-3, atol=2e-3,
    )


class MixedPrecisionLayer(torch.nn.Module):
    """Use FP16 parameters and keep the output in FP32 for the training loss."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.75, dtype=torch.float16))

    def forward(self, state, inputs):
        """Return the scaled state plus the injected input as FP32."""
        return (self.scale * state + inputs).float()


def test_mixed_precision_training_matches_scalar_reference():
    """Real SDK solves, implicit backward and SGD work with a finite regularizer."""
    layer = MixedPrecisionLayer()
    config = QDEQConfig(
        num_basis=1, coeff_bits=2, f_thres=20, b_thres=80,
        f_eps=1e-4, b_eps=1e-3, fd_eps=0.1, jac_loss_vecs=1,
    )
    model = QDEQ(layer, BruteForceOptimizer(), config)
    inputs = torch.full((8, 128, 128), 0.25, dtype=torch.float16, requires_grad=True)
    warm = torch.ones_like(inputs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    previous_loss = float("inf")
    draw_probe = torch.randn_like
    probes = []

    def record_probe(state):
        probe = draw_probe(state)
        probes.append(probe)
        return probe

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        for step in range(3):
            scale_before = layer.scale.item()
            optimizer.zero_grad()
            inputs.grad = None
            probes.clear()

            with mock.patch("kaiwu.torch_plugin.qdeq.torch.randn_like", side_effect=record_probe):
                output = model(inputs, warm)
            penalty = model.last_jac_loss
            loss = output.square().mean() + 0.1 * penalty
            assert torch.isfinite(loss)
            assert loss.item() < previous_loss
            assert penalty.dtype == torch.float32

            equilibrium = 0.25 / (1 - scale_before)
            torch.testing.assert_close(
                output, torch.full_like(output, equilibrium), rtol=3e-4, atol=3e-4,
            )
            probe_energy = probes[0].double().square().mean().item()
            assert penalty.item() == pytest.approx(scale_before**2 * probe_energy, rel=1e-3)
            loss.backward()
            expected_gradient = (
                2 * equilibrium**2 / (1 - scale_before)
                + 0.2 * scale_before * probe_energy
            )
            assert layer.scale.grad.item() == pytest.approx(expected_gradient, rel=1e-2)
            input_gradient = 2 * equilibrium / ((1 - scale_before) * inputs.numel())
            torch.testing.assert_close(
                inputs.grad.double(), torch.full_like(inputs.double(), input_gradient),
                rtol=1e-2, atol=1e-7,
            )
            stats = model.get_last_stats()
            assert stats["forward_residual"] <= config.f_eps
            assert stats["backward_residual"] <= config.b_eps
            if step > 0:
                assert stats["forward_sampler_time"] > 0
            optimizer.step()
            assert layer.scale.item() < scale_before
            print(json.dumps({
                "step": step + 1, "loss": loss.item(), "jacobian_penalty": penalty.item(),
                "scale_before": scale_before, "scale_after": layer.scale.item(),
                "scale_gradient": layer.scale.grad.item(),
                "equilibrium_reference": equilibrium, "equilibrium_mean": output.mean().item(),
                "forward_residual": stats["forward_residual"],
                "backward_residual": stats["backward_residual"],
            }))
            previous_loss = loss.item()
            warm = output.detach().to(torch.float16)
