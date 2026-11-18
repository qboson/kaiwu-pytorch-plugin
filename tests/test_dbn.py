"""Public interface tests for ``UnsupervisedDBN``.

The class is exported from ``kaiwu.torch_plugin`` but had no tests; these pin the
documented contracts, including the "model not built yet" accessors that
``example/dbn_digits/dbn_trainer.py`` relies on
(``if dbn.num_layers == 0: dbn.create_rbm_layer(...)``).
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import UnsupervisedDBN

# 3 visible -> 2 hidden weights and the combined [visible(3), hidden(2)] bias
WEIGHTS = torch.tensor([[1.0, -2.0], [0.5, 0.0], [-1.0, 1.0]])
BIAS = torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5])


def _make_trained_dbn() -> UnsupervisedDBN:
    """A one-layer DBN with deterministic weights."""
    dbn = UnsupervisedDBN([2]).create_rbm_layer(3)
    rbm = dbn.get_rbm_layer(0)
    with torch.no_grad():
        rbm.quadratic_coef.copy_(WEIGHTS)
        rbm.linear_bias.copy_(BIAS)
    return dbn.mark_as_trained()


class TestUnsupervisedDBN(unittest.TestCase):
    """Construction, accessors, feature extraction and reconstruction."""

    def test_defaults(self):
        """Defaults match the documented hidden-layer structure."""
        dbn = UnsupervisedDBN()
        self.assertEqual(dbn.hidden_layers_structure, [100, 100])
        self.assertIsNone(dbn.rbm_layers)
        self.assertIsNone(dbn.input_dim)

    def test_accessors_are_safe_before_building(self):
        """Unbuilt accessors must not raise (num_layers == 0 is the documented guard)."""
        dbn = UnsupervisedDBN([5, 4])
        self.assertEqual(dbn.num_layers, 0)
        self.assertIsNone(dbn.output_dim)
        self.assertIsNone(dbn.get_rbm_layer(0))

    def test_create_rbm_layer_builds_the_documented_stack(self):
        """Layers chain visible -> hidden -> hidden with the right sizes."""
        dbn = UnsupervisedDBN([5, 4]).create_rbm_layer(7)
        self.assertEqual(dbn.input_dim, 7)
        self.assertEqual(dbn.num_layers, 2)
        self.assertEqual(dbn.output_dim, 4)
        self.assertEqual(
            [(rbm.num_visible, rbm.num_hidden) for rbm in dbn.rbm_layers],
            [(7, 5), (5, 4)],
        )
        self.assertIsNone(dbn.get_rbm_layer(2))
        self.assertIs(dbn.get_rbm_layer(-1), dbn.rbm_layers[-1])

    def test_forward_requires_a_built_and_trained_model(self):
        """forward/transform gate on build state and training readiness."""
        with self.assertRaises(ValueError):
            UnsupervisedDBN([3]).forward(np.zeros((2, 4), dtype=np.float32))

        dbn = UnsupervisedDBN([3]).create_rbm_layer(4)
        with self.assertRaises(ValueError):
            dbn.forward(np.zeros((2, 4), dtype=np.float32))

        dbn.mark_as_trained()
        inputs = np.zeros((2, 4), dtype=np.float32)
        features = dbn.forward(inputs)
        self.assertEqual(features.shape, (2, 3))
        np.testing.assert_allclose(dbn.transform(inputs), features)

    def test_forward_matches_hand_computed_hidden_probabilities(self):
        """Feature extraction returns sigmoid(x W + b_hidden)."""
        dbn = _make_trained_dbn()
        inputs = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        expected = 1.0 / (
            1.0 + np.exp(-(inputs @ WEIGHTS.numpy() + BIAS.numpy()[3:]))
        )
        np.testing.assert_allclose(dbn.forward(inputs), expected, rtol=1e-6)

    def test_reconstruct_returns_the_documented_pair(self):
        """reconstruct() yields (reconstruction, per-sample MSE) as documented."""
        dbn = _make_trained_dbn()
        inputs = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        reconstruction, errors = dbn.reconstruct(inputs)

        hidden = 1.0 / (
            1.0 + np.exp(-(inputs @ WEIGHTS.numpy() + BIAS.numpy()[3:]))
        )
        expected = 1.0 / (
            1.0 + np.exp(-(hidden @ WEIGHTS.numpy().T + BIAS.numpy()[:3]))
        )
        np.testing.assert_allclose(reconstruction, expected, rtol=1e-6)
        np.testing.assert_allclose(errors, ((inputs - expected) ** 2).mean(axis=1), rtol=1e-6)

    def test_reconstruct_validates_the_layer_index(self):
        """Out-of-range layer indexes fail with the documented ValueError."""
        dbn = UnsupervisedDBN([3]).create_rbm_layer(2)
        with self.assertRaises(ValueError):
            dbn.reconstruct(np.zeros((1, 2), dtype=np.float32), layer_index=1)

    def test_reconstruct_without_layers_fails(self):
        """A model with no layers reports the documented ValueError."""
        with self.assertRaises(ValueError):
            UnsupervisedDBN([3]).reconstruct(np.zeros((1, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
