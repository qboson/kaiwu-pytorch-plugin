"""GBRBM must follow parameter devices when ``self.device`` is stale.

``self.device`` is a plain attribute refreshed only when the module is moved.
Replacing parameters, or any other update that does not go through ``_apply``,
leaves the cache pointing at a different device. Buffers allocated from that
cache are then empty meta tensors or raise a device mismatch. These tests
keep the weights on CPU and point the cache at ``meta``.
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


def _weighted_model():
    """Return a small deterministic float64 GBRBM on CPU."""
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=2,
        num_hidden=2,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        model.mu.copy_(torch.tensor([1.0, -1.0], dtype=torch.float64))
        model.log_var.copy_(torch.log(torch.tensor([4.0, 1.0], dtype=torch.float64)))
        model.quadratic_coef.copy_(
            torch.tensor([[2.0, -1.0], [0.5, 1.0]], dtype=torch.float64)
        )
        model.linear_bias.copy_(torch.tensor([0.25, -0.5], dtype=torch.float64))
    return model


def _stale_cache(model):
    """Copy ``model`` and point its device cache away from the parameters."""
    stale = _weighted_model()
    stale.load_state_dict(model.state_dict())
    stale.device = torch.device("meta")
    return stale


class _FixedSampler:
    """Return the same two Bernoulli spins on every call."""

    def solve(self, ising_mat):
        self.ising_shape = ising_mat.shape
        return np.array([[1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=np.float32)


class TestGaussianBernoulliDevice(unittest.TestCase):
    """Runtime tensors follow the parameters, not the cached device."""

    def setUp(self):
        self.reference = _weighted_model()
        self.stale = _stale_cache(self.reference)
        self.s_gaussian = torch.tensor(
            [[3.0, 0.0], [1.0, -2.0]],
            dtype=torch.float64,
        )
        self.s_bernoulli = torch.tensor(
            [[1.0, 0.0], [1.0, 1.0]],
            dtype=torch.float64,
        )
        self.s_all = torch.tensor(
            [[3.0, 0.0, 1.0, 0.0], [1.0, -2.0, 0.0, 1.0]],
            dtype=torch.float64,
        )

    def test_parameters_stay_on_cpu_when_the_cache_is_meta(self):
        """The cache is the only thing that was moved."""
        self.assertEqual(self.stale.device, torch.device("meta"))
        self.assertEqual(self.stale.quadratic_coef.device, torch.device("cpu"))

    def test_ising_and_conditional_states_stay_on_the_parameter_device(self):
        """Ising conversion and both conditionals must return real CPU values."""
        np.testing.assert_allclose(
            self.stale.get_ising_matrix(),
            self.reference.get_ising_matrix(),
        )

        from_gaussian = self.stale.infer_from_gaussian(
            self.s_gaussian,
            binarize=False,
        )
        self.assertEqual(from_gaussian.device, torch.device("cpu"))
        torch.testing.assert_close(
            from_gaussian,
            self.reference.infer_from_gaussian(self.s_gaussian, binarize=False),
        )

        from_bernoulli = self.stale.infer_from_bernoulli(
            self.s_bernoulli,
            no_random=True,
        )
        self.assertEqual(from_bernoulli.device, torch.device("cpu"))
        torch.testing.assert_close(
            from_bernoulli,
            self.reference.infer_from_bernoulli(self.s_bernoulli, no_random=True),
        )

    def test_gibbs_and_positive_phase_ignore_the_stale_cache(self):
        """Gibbs draws and the positive phase must match a coherent model."""
        torch.manual_seed(0)
        expected_gibbs = self.reference.gibbs_sample(
            n_step=2,
            n_burnin=0,
            s_gaussian=self.s_gaussian,
        )
        torch.manual_seed(0)
        actual_gibbs = self.stale.gibbs_sample(
            n_step=2,
            n_burnin=0,
            s_gaussian=self.s_gaussian,
        )

        self.assertEqual(actual_gibbs.device, torch.device("cpu"))
        torch.testing.assert_close(actual_gibbs, expected_gibbs)

        self.reference.zero_grad()
        self.stale.zero_grad()
        self.reference.positive_phase_energy_expectation(
            self.s_gaussian
        ).sum().backward()
        self.stale.positive_phase_energy_expectation(self.s_gaussian).sum().backward()
        self.assertEqual(self.stale.quadratic_coef.grad.device, torch.device("cpu"))
        torch.testing.assert_close(
            self.stale.quadratic_coef.grad,
            self.reference.quadratic_coef.grad,
        )

    def test_energy_values_ignore_the_stale_cache(self):
        """Energy reads the parameters, so a stale cache must not change it."""
        torch.testing.assert_close(
            self.stale.energy(self.s_all),
            self.reference.energy(self.s_all),
        )
        torch.testing.assert_close(
            self.stale.marginal_energy(self.s_gaussian),
            self.reference.marginal_energy(self.s_gaussian),
        )
        self.reference.zero_grad()
        self.reference.energy(self.s_all, enable_grad=True).sum().backward()
        self.assertIsNotNone(self.reference.quadratic_coef.grad)
        self.assertTrue(torch.isfinite(self.reference.quadratic_coef.grad).all())

    def test_sampler_output_stays_on_the_parameter_device(self):
        """External samples must not be staged on the stale device cache."""
        expected = self.reference.sample(_FixedSampler())
        actual = self.stale.sample(_FixedSampler())

        self.assertEqual(actual.device, torch.device("cpu"))
        torch.testing.assert_close(actual, expected)

        torch.manual_seed(0)
        expected_gibbs = self.reference.gibbs_sample(
            n_step=1,
            n_burnin=0,
            sampler=_FixedSampler(),
        )
        torch.manual_seed(0)
        actual_gibbs = self.stale.gibbs_sample(
            n_step=1,
            n_burnin=0,
            sampler=_FixedSampler(),
        )
        self.assertEqual(actual_gibbs.device, torch.device("cpu"))
        torch.testing.assert_close(actual_gibbs, expected_gibbs)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGaussianBernoulliCudaInputs(unittest.TestCase):
    """CPU caller tensors are scored on the parameter device."""

    def test_cpu_states_follow_cuda_parameters_when_cache_is_cpu(self):
        model = GaussianBernoulliRestrictedBoltzmannMachine(num_visible=2, num_hidden=2)
        self.assertEqual(model.quadratic_coef.device.type, "cuda")
        model.device = torch.device("cpu")
        states = torch.randn(4, model.num_nodes)
        gaussian = states[:, : model.num_gaussian]

        energy = model.energy(states)
        inferred = model.infer_from_gaussian(gaussian, binarize=False)
        sampled = model.gibbs_sample(n_step=1, n_burnin=0, s_gaussian=gaussian)

        self.assertEqual(energy.device.type, "cuda")
        self.assertEqual(inferred.device.type, "cuda")
        self.assertEqual(sampled.device.type, "cuda")
        self.assertTrue(torch.isfinite(energy).all())


if __name__ == "__main__":
    unittest.main()
