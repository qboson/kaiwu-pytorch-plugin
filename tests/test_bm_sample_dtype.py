"""Regression tests for the dtype of sampler-backed BM samples.

``AbstractBoltzmannMachine.sample()`` used to build its return value with
``torch.FloatTensor``, so a model whose parameters are ``torch.float64``
still received ``float32`` samples. The returned samples must use the model
parameter dtype (and the model device) while preserving the existing
spin-to-binary decoding.
"""

import unittest

import numpy as np
import torch

from kaiwu.torch_plugin import BoltzmannMachine


class SpinRowSampler:
    """Deterministic stand-in for a Kaiwu optimizer returning spin rows."""

    def solve(self, ising_mat):
        rows = np.ones((2, ising_mat.shape[0]))
        # Alternate the second row: a -1 auxiliary spin plus -1 variable
        # spins at even indices makes the decoded row cover both binary
        # values.
        rows[1, -1] = -1.0
        rows[1, :-1:2] = -1.0
        return rows


class TestSampleDtype(unittest.TestCase):
    """Sampler-backed samples must follow the model parameter dtype."""

    def test_constructor_parameters_define_sample_dtype(self):
        """A model built from float64 tensors must sample in float64."""
        bm = BoltzmannMachine(
            4,
            quadratic_coef=torch.ones((4, 4), dtype=torch.float64),
            linear_bias=torch.zeros(4, dtype=torch.float64),
            device="cpu",
        )

        samples = bm.sample(SpinRowSampler())

        self.assertEqual(samples.dtype, torch.float64)
        self.assertEqual(samples.device, bm.quadratic_coef.device)

    def test_dtype_move_changes_sample_dtype(self):
        """``.to(dtype=...)`` must move the sample dtype along with params."""
        bm = BoltzmannMachine(4, device="cpu")
        bm.to(dtype=torch.float64)

        samples = bm.sample(SpinRowSampler())

        self.assertEqual(samples.dtype, torch.float64)
        self.assertEqual(samples.device, bm.quadratic_coef.device)

    def test_spin_to_binary_decoding_is_preserved(self):
        """The dtype fix must not change the spin-to-binary decoding."""
        bm = BoltzmannMachine(4, device="cpu")

        samples = bm.sample(SpinRowSampler())

        self.assertEqual(samples.dtype, torch.float32)
        self.assertEqual(samples.shape, (2, 4))
        self.assertTrue(torch.all((samples == 0) | (samples == 1)))
        self.assertEqual(samples[0].tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(samples[1].tolist(), [1.0, 0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
