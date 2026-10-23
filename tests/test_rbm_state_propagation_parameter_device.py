"""Regression tests for RBM state propagation on the parameter device.

``RestrictedBoltzmannMachine.get_hidden`` and ``get_visible`` allocate their
working buffers with the cached ``self.device`` attribute. After a parent module
moves only the RBM parameters (via ``_apply``), that cache can stay on CPU while
the weights live on another device, so assigning caller-visible states into the
buffer raises a cross-device ``RuntimeError``.
"""

import unittest

import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


class TestRbmStatePropagationParameterDevice(unittest.TestCase):
    """State propagation must follow parameter placement, not a stale cache."""

    def test_get_hidden_follows_parameter_device_after_apply_move(self):
        """Visible states must land on the same device as the RBM weights."""
        rbm = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2, device="cpu")
        rbm._apply(lambda tensor: tensor.to(torch.device("meta")))

        visible = torch.ones(4, 3)
        hidden_state = rbm.get_hidden(visible)

        self.assertEqual(hidden_state.device, rbm.quadratic_coef.device)
        self.assertEqual(hidden_state.shape, (4, 5))

    def test_get_visible_follows_parameter_device_after_apply_move(self):
        """Hidden states must land on the same device as the RBM weights."""
        rbm = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2, device="cpu")
        rbm._apply(lambda tensor: tensor.to(torch.device("meta")))

        hidden = torch.ones(4, 2)
        visible_state = rbm.get_visible(hidden)

        self.assertEqual(visible_state.device, rbm.quadratic_coef.device)
        self.assertEqual(visible_state.shape, (4, 5))


if __name__ == "__main__":
    unittest.main()
