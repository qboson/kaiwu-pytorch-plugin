"""dtype contract tests for RBM state propagation.

``RestrictedBoltzmannMachine.get_hidden`` and ``get_visible`` allocate their
full-state buffer with ``torch.zeros`` defaults, so a model converted with
``double()`` or ``half()`` silently returned float32 states. The tests below
pin the contract that the returned states follow the model's parameter dtype.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


class TestRBMStateDtype(unittest.TestCase):
    """State propagation must keep the precision of the model parameters."""

    def _reference_hidden(self, rbm, s_visible):
        batch = s_visible.size(0)
        expected = torch.zeros(
            batch, rbm.num_visible + rbm.num_hidden, dtype=s_visible.dtype
        )
        expected[:, : rbm.num_visible] = s_visible
        expected[:, rbm.num_visible :] = torch.sigmoid(
            s_visible @ rbm.quadratic_coef + rbm.linear_bias[rbm.num_visible :]
        )
        return expected

    def _reference_visible(self, rbm, s_hidden):
        batch = s_hidden.size(0)
        expected = torch.zeros(
            batch, rbm.num_visible + rbm.num_hidden, dtype=s_hidden.dtype
        )
        expected[:, rbm.num_visible :] = s_hidden
        expected[:, : rbm.num_visible] = torch.sigmoid(
            s_hidden @ rbm.quadratic_coef.t() + rbm.linear_bias[: rbm.num_visible]
        )
        return expected

    def test_get_hidden_preserves_double_dtype(self):
        rbm = RestrictedBoltzmannMachine(4, 3).double()
        s_visible = torch.randn(2, 4, dtype=torch.float64)

        result = rbm.get_hidden(s_visible)

        self.assertEqual(result.dtype, torch.float64)
        self.assertTrue(torch.equal(result, self._reference_hidden(rbm, s_visible)))

    def test_get_visible_preserves_double_dtype(self):
        rbm = RestrictedBoltzmannMachine(4, 3).double()
        s_hidden = torch.randn(2, 3, dtype=torch.float64)

        result = rbm.get_visible(s_hidden)

        self.assertEqual(result.dtype, torch.float64)
        self.assertTrue(torch.equal(result, self._reference_visible(rbm, s_hidden)))

    def test_get_hidden_preserves_half_dtype(self):
        rbm = RestrictedBoltzmannMachine(4, 3).half()
        s_visible = torch.randn(2, 4, dtype=torch.float16)

        result = rbm.get_hidden(s_visible)

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape, (2, 7))

    def test_bernoulli_sampling_follows_model_dtype(self):
        rbm = RestrictedBoltzmannMachine(4, 3).double()
        s_visible = torch.randn(2, 4, dtype=torch.float64)

        hidden = rbm.get_hidden(s_visible, bernoulli=True)
        visible = rbm.get_visible(torch.randn(2, 3, dtype=torch.float64), bernoulli=True)

        self.assertEqual(hidden.dtype, torch.float64)
        self.assertEqual(visible.dtype, torch.float64)
        # The newly drawn states must be binary.
        self.assertTrue(
            torch.isin(
                hidden[:, rbm.num_visible :],
                torch.tensor([0.0, 1.0], dtype=torch.float64),
            ).all()
        )
        self.assertTrue(
            torch.isin(
                visible[:, : rbm.num_visible],
                torch.tensor([0.0, 1.0], dtype=torch.float64),
            ).all()
        )

    def test_float32_behavior_is_unchanged(self):
        """The default float32 path must produce exactly the previous values."""
        torch.manual_seed(7)
        rbm = RestrictedBoltzmannMachine(5, 4)
        s_visible = torch.randn(3, 5)

        hidden = rbm.get_hidden(s_visible)

        self.assertEqual(hidden.dtype, torch.float32)
        self.assertTrue(
            torch.equal(hidden, self._reference_hidden(rbm, s_visible))
        )

    def test_double_states_feed_forward_without_downcast(self):
        """End-to-end: converted model energy no longer mixes float32 states."""
        rbm = RestrictedBoltzmannMachine(4, 3).double()
        s_visible = torch.randn(2, 4, dtype=torch.float64)

        state = rbm.get_hidden(s_visible)
        energy = rbm(state)

        self.assertEqual(state.dtype, torch.float64)
        self.assertEqual(energy.dtype, torch.float64)
        self.assertTrue(torch.isfinite(energy).all())


if __name__ == "__main__":
    unittest.main()
