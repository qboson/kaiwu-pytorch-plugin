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

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.dbn import UnsupervisedDBN


class DummySampler:
    """Return fixed Ising spin solutions with a trailing gauge spin."""

    def solve(self, ising_mat):
        """Return one all-positive spin row plus the gauge column."""
        n = ising_mat.shape[0]
        return np.ones((2, n), dtype=np.float32)


class TestRestrictedBoltzmannMachineDtype(unittest.TestCase):
    """dtype consistency of the plain RBM, in the spirit of the GBRBM fix."""

    def test_dtype_argument_builds_consistent_parameters(self):
        rbm = RestrictedBoltzmannMachine(
            4,
            3,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.assertEqual(rbm.quadratic_coef.dtype, torch.float64)
        self.assertEqual(rbm.linear_bias.dtype, torch.float64)
        energies = rbm.forward(torch.ones(2, 7, dtype=torch.float64))
        self.assertEqual(energies.dtype, torch.float64)

    def test_supplied_weights_are_coerced_to_dtype(self):
        # Mixed-dtype supplied weights used to leave the model parameters in
        # float64 and float32, so every forward pass raised a RuntimeError.
        rbm = RestrictedBoltzmannMachine(
            4,
            3,
            quadratic_coef=torch.randn(4, 3, dtype=torch.float64),
            linear_bias=torch.zeros(7, dtype=torch.float32),
            device=torch.device("cpu"),
        )
        self.assertEqual(rbm.quadratic_coef.dtype, torch.float32)
        self.assertEqual(rbm.linear_bias.dtype, torch.float32)
        energies = rbm.forward(torch.ones(2, 7))
        self.assertEqual(energies.dtype, torch.float32)

    def test_supplied_float32_weights_with_float64_dtype(self):
        rbm = RestrictedBoltzmannMachine(
            4,
            3,
            quadratic_coef=torch.randn(4, 3),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.assertEqual(rbm.quadratic_coef.dtype, torch.float64)
        self.assertEqual(rbm.linear_bias.dtype, torch.float64)

    def test_supplied_float64_weights_with_float64_dtype(self):
        rbm = RestrictedBoltzmannMachine(
            4,
            3,
            quadratic_coef=torch.randn(4, 3, dtype=torch.float64),
            linear_bias=torch.zeros(7, dtype=torch.float64),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.assertEqual(rbm.quadratic_coef.dtype, torch.float64)
        self.assertEqual(rbm.linear_bias.dtype, torch.float64)
        energies = rbm.forward(torch.ones(2, 7, dtype=torch.float64))
        self.assertEqual(energies.dtype, torch.float64)

    def test_double_model_sampling_chain(self):
        # .double() used to break the model's own sampling chain: get_hidden
        # and get_visible downcast the state to float32 and the next forward
        # pass crashed with a dtype mismatch.
        rbm = RestrictedBoltzmannMachine(4, 3, device=torch.device("cpu")).double()
        self.assertEqual(rbm.dtype, torch.float64)
        states = rbm.get_hidden(torch.ones(2, 4, dtype=torch.float64))
        self.assertEqual(states.dtype, torch.float64)
        energies = rbm.forward(states)
        self.assertEqual(energies.dtype, torch.float64)
        visible = rbm.get_visible(torch.ones(2, 3, dtype=torch.float64))
        self.assertEqual(visible.dtype, torch.float64)

    def test_dtype_syncs_through_to_and_nested_move(self):
        # .to(dtype) (and .double(), and moves initiated by a parent module)
        # must keep self.dtype in sync, the same way self.device already is.
        rbm = RestrictedBoltzmannMachine(4, 3, device=torch.device("cpu"))
        rbm.to(torch.float64)
        self.assertEqual(rbm.dtype, torch.float64)
        states = rbm.get_hidden(torch.ones(2, 4, dtype=torch.float64))
        self.assertEqual(states.dtype, torch.float64)
        energies = rbm.forward(states)
        self.assertEqual(energies.dtype, torch.float64)

        parent = torch.nn.Sequential()
        parent.add_module("rbm", RestrictedBoltzmannMachine(4, 3))
        parent.to(torch.float64)
        self.assertEqual(parent.rbm.dtype, torch.float64)

    def test_sample_returns_model_dtype(self):
        # sample() used to hardcode torch.FloatTensor, so a float64 model
        # received float32 states back from its own sampler.
        rbm = RestrictedBoltzmannMachine(
            4,
            3,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        samples = rbm.sample(DummySampler())
        self.assertEqual(samples.dtype, torch.float64)
        # The sampled states must be usable in the model's own forward pass.
        energies = rbm.forward(samples)
        self.assertEqual(energies.dtype, torch.float64)

    def test_ising_matrix_matches_model_dtype(self):
        rbm = RestrictedBoltzmannMachine(
            4,
            3,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        ising_mat = rbm.get_ising_matrix()
        self.assertEqual(ising_mat.dtype, np.float64)

    def test_default_dtype_stays_float32(self):
        rbm = RestrictedBoltzmannMachine(4, 3, device=torch.device("cpu"))
        self.assertEqual(rbm.quadratic_coef.dtype, torch.float32)
        self.assertEqual(rbm.linear_bias.dtype, torch.float32)


class TestBoltzmannMachineDtype(unittest.TestCase):
    """dtype consistency of the full Boltzmann Machine."""

    def test_dtype_argument_builds_consistent_parameters(self):
        bm = BoltzmannMachine(6, dtype=torch.float64, device=torch.device("cpu"))
        self.assertEqual(bm.quadratic_coef.dtype, torch.float64)
        self.assertEqual(bm.linear_bias.dtype, torch.float64)
        energies = bm.forward(torch.ones(2, 6, dtype=torch.float64))
        self.assertEqual(energies.dtype, torch.float64)

    def test_supplied_mixed_weights_are_coerced(self):
        bm = BoltzmannMachine(
            6,
            quadratic_coef=torch.randn(6, 6, dtype=torch.float64),
            linear_bias=torch.zeros(6, dtype=torch.float32),
            device=torch.device("cpu"),
        )
        self.assertEqual(bm.quadratic_coef.dtype, torch.float32)
        self.assertEqual(bm.linear_bias.dtype, torch.float32)
        energies = bm.forward(torch.ones(2, 6))
        self.assertEqual(energies.dtype, torch.float32)

    def test_gibbs_sample_on_double_model(self):
        # gibbs_sample used to crash on a float64 model because the state
        # tensor was always built in float32.
        bm = BoltzmannMachine(6, dtype=torch.float64, device=torch.device("cpu"))
        samples = bm.gibbs_sample(num_steps=3, num_sample=4)
        self.assertEqual(samples.dtype, torch.float64)
        self.assertEqual(samples.shape, (4, 6))
        samples = bm.gibbs_sample(
            num_steps=3,
            s_visible=torch.ones(2, 2, dtype=torch.float64),
        )
        self.assertEqual(samples.dtype, torch.float64)

    def test_condition_sample_follows_model_dtype(self):
        bm = BoltzmannMachine(6, dtype=torch.float64, device=torch.device("cpu"))
        samples = bm.condition_sample(
            DummySampler(), torch.ones(2, 2, dtype=torch.float64)
        )
        self.assertEqual(samples.dtype, torch.float64)

    def test_condition_sample_explicit_dtype(self):
        bm = BoltzmannMachine(6, dtype=torch.float64, device=torch.device("cpu"))
        samples = bm.condition_sample(
            DummySampler(), torch.ones(2, 2, dtype=torch.float64), dtype=torch.float32
        )
        self.assertEqual(samples.dtype, torch.float32)

    def test_sample_returns_model_dtype(self):
        bm = BoltzmannMachine(
            6,
            quadratic_coef=torch.randn(6, 6, dtype=torch.float64),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        samples = bm.sample(DummySampler())
        self.assertEqual(samples.dtype, torch.float64)


class TestUnsupervisedDBNDtype(unittest.TestCase):
    """The DBN helpers used to force float32 on every input."""

    def _float64_dbn(self):
        dbn = UnsupervisedDBN([3, 2])
        dbn.create_rbm_layer(4)
        for i, rbm in enumerate(dbn.rbm_layers):
            dbn.rbm_layers[i] = RestrictedBoltzmannMachine(
                rbm.num_visible,
                rbm.num_hidden,
                dtype=torch.float64,
                device=torch.device("cpu"),
            )
        dbn.device = torch.device("cpu")
        dbn.mark_as_trained()
        return dbn

    def test_transform_honours_layer_dtype(self):
        dbn = self._float64_dbn()
        features = dbn.transform(np.ones((5, 4), dtype=np.float64))
        self.assertEqual(features.dtype, np.float64)
        self.assertEqual(features.shape, (5, 2))

    def test_float32_input_still_supported(self):
        dbn = UnsupervisedDBN([3])
        dbn.create_rbm_layer(4)
        dbn.mark_as_trained()
        features = dbn.transform(np.ones((5, 4), dtype=np.float32))
        self.assertEqual(features.dtype, np.float32)

    def test_reconstruct_with_rbm_honours_dtype(self):
        rbm = RestrictedBoltzmannMachine(
            4,
            3,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        visible_recon, errors = UnsupervisedDBN.reconstruct_with_rbm(
            rbm, np.ones((5, 4), dtype=np.float64)
        )
        self.assertEqual(visible_recon.dtype, np.float64)
        self.assertEqual(errors.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
