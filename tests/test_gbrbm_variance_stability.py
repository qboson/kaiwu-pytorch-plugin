"""Numerical stability tests for the GBRBM variance guard.

``GaussianBernoulliRestrictedBoltzmannMachine.var`` clipped the variance only
from below. When ``log_var`` grows past the float32 exponent range the
variance becomes ``inf``, the energy loses all dependence on the Gaussian
state, the precision ``1 / var`` underflows to zero, and backpropagation
produces ``NaN`` gradients (``0 * inf`` in the exp backward), which silently
poisons every subsequent optimizer step. The tests below pin the both-sided
guard.
"""

import os
import sys
import unittest

import numpy as np
import torch

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

import kaiwu

# Extend namespace package __path__ so that local src is preferred
kaiwu.__path__ = list(kaiwu.__path__) + [os.path.join(src_root, "kaiwu")]

for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


class TestGBRBMVarianceStability(unittest.TestCase):
    """The variance must stay finite and keep gradients finite."""

    def _model_with_log_var(self, value, num_gaussian=3, num_bernoulli=2):
        model = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=num_gaussian,
            num_hidden=num_bernoulli,
            is_visible_gaussian=True,
        )
        with torch.no_grad():
            model.log_var.fill_(float(value))
            model.mu.fill_(0.25)
            model.quadratic_coef.fill_(0.5)
            model.linear_bias.fill_(0.1)
        return model

    def test_saturated_variance_keeps_gradients_finite(self):
        model = self._model_with_log_var(100.0)
        state = torch.tensor([[1.0, -0.5, 2.0, 1.0, 0.0]])

        energy = model.energy(state, enable_grad=True).sum()
        energy.backward()

        self.assertTrue(torch.isfinite(energy).all())
        for name, parameter in model.named_parameters():
            self.assertTrue(
                torch.isfinite(parameter.grad).all(),
                f"{name}.grad contains non-finite values: {parameter.grad}",
            )

    def test_energy_keeps_gaussian_dependence_under_saturated_variance(self):
        """With var = inf the energy collapsed to a constant on the Gaussian state."""
        model = self._model_with_log_var(100.0)
        state_a = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
        state_b = torch.tensor([[-50.0, 30.0, 10.0, 1.0, 0.0]])

        energy_a = model.energy(state_a)
        energy_b = model.energy(state_b)

        self.assertNotEqual(
            energy_a.item(), energy_b.item(),
            "energy must keep depending on the Gaussian state",
        )
        self.assertTrue(torch.isfinite(energy_a).all())
        self.assertTrue(torch.isfinite(energy_b).all())

    def test_variance_is_bounded_on_both_sides(self):
        large = self._model_with_log_var(100.0)
        small = self._model_with_log_var(-100.0)

        self.assertTrue(torch.isfinite(large.var).all())
        # exp(log(1/eps)) can round one ulp above 1/eps in float32.
        self.assertLessEqual(float(large.var.max()), (1.0 / large.eps) * 1.001)
        self.assertGreaterEqual(float(small.var.min()), small.eps * 0.999)
        self.assertTrue(torch.isfinite(large.std).all())
        self.assertTrue(torch.isfinite(small.std).all())

    def test_precision_stays_nonzero_for_ising_conversion(self):
        model = self._model_with_log_var(100.0)

        precision = model.diag_precision
        self.assertTrue(torch.isfinite(precision).all())
        self.assertTrue((precision.diag() > 0).all())

        ising = model._to_ising_matrix()  # pylint: disable=protected-access
        self.assertTrue(np.isfinite(ising).all())
        self.assertTrue((ising != 0).any())

    def test_inference_stays_finite_under_saturated_variance(self):
        model = self._model_with_log_var(100.0)
        gaussian = torch.tensor([[1.0, -1.0, 2.0]])

        inferred = model.infer_from_gaussian(gaussian)
        restored = model.infer_from_bernoulli(
            torch.tensor([[1.0, 0.0]]), no_random=True
        )

        self.assertTrue(torch.isfinite(inferred).all())
        self.assertTrue(torch.isfinite(restored).all())

    def test_unclipped_range_is_bitwise_unchanged(self):
        """Inside the guard bounds the variance, energy and grads are unchanged."""
        for log_var_value in (-3.0, -1.0, 0.0, 2.0, 10.0):
            with self.subTest(log_var=log_var_value):
                model = self._model_with_log_var(log_var_value)
                state = torch.tensor([[1.0, -0.5, 2.0, 1.0, 0.0]])

                var_guarded = model.var
                var_direct = model.log_var.exp().clip(min=model.eps)
                self.assertTrue(torch.equal(var_guarded, var_direct))

                energy = model.energy(state, enable_grad=True).sum()
                energy.backward()
                guarded_grad = model.log_var.grad.clone()

                # Recompute the energy with the unguarded formula. The variance
                # property is evaluated once per term, so the reference mirrors
                # that two-node graph to keep the comparison bitwise.
                log_var = model.log_var.detach().clone().requires_grad_(True)
                s_gaussian = state[:, : model.num_gaussian]
                s_bernoulli = state[:, model.num_gaussian :]
                reference_energy = (
                    0.5
                    * torch.sum(
                        (s_gaussian - model.mu).square()
                        / log_var.exp().clip(min=model.eps),
                        dim=-1,
                    )
                    - torch.sum(
                        (s_gaussian / log_var.exp().clip(min=model.eps))
                        @ model.quadratic_coef
                        * s_bernoulli,
                        dim=-1,
                    )
                    - s_bernoulli @ model.linear_bias
                ).sum()
                reference_energy.backward()

                self.assertTrue(
                    torch.equal(guarded_grad, log_var.grad),
                    "gradients must be bitwise identical inside the guard bounds",
                )

    def test_marginal_energy_stays_finite_under_saturation(self):
        model = self._model_with_log_var(100.0)
        gaussian = torch.tensor([[1.0, -1.0, 2.0]])

        marginal = model.marginal_energy(gaussian)
        expected = model.positive_phase_energy_expectation(
            gaussian, enable_grad=False
        )

        self.assertTrue(torch.isfinite(marginal).all())
        self.assertTrue(torch.isfinite(expected).all())


if __name__ == "__main__":
    unittest.main()
