"""Regression tests for full BoltzmannMachine forward device alignment.

Open #1268 already coerces inputs on RBM ``forward`` and on sampling helpers;
the full ``BoltzmannMachine.forward`` Hamiltonian path was still mixing caller
CPU tensors with parameters moved via ``_apply`` (the #164 parent-move
scenario). These tests follow the meta-device pattern from
``tests/test_bm_device_sync.py`` so they run without a GPU.
"""

import unittest

import torch
from torch import nn

from kaiwu.torch_plugin import BoltzmannMachine


class TestBoltzmannMachineForwardParameterDevice(unittest.TestCase):
    """``forward`` must follow live parameters, not a stale ``self.device``."""

    def test_forward_after_apply_move_accepts_cpu_batch(self):
        """CPU caller tensors must work once parameters were moved via ``_apply``."""
        bm = BoltzmannMachine(4, device="cpu")
        bm._apply(lambda tensor: tensor.to(torch.device("meta")))

        spins = torch.ones(2, 4)
        energy = bm(spins)

        self.assertEqual(energy.shape, (2,))
        self.assertEqual(energy.device, torch.device("meta"))

    def test_forward_after_parent_move_accepts_cpu_batch(self):
        """Nested modules must score without forcing callers onto the accelerator."""
        parent = nn.Module()
        parent.bm = BoltzmannMachine(4, device="cpu")
        parent._apply(lambda tensor: tensor.to(torch.device("meta")))

        energy = parent.bm(torch.zeros(3, 4))

        self.assertEqual(energy.shape, (3,))
        self.assertEqual(energy.device, torch.device("meta"))


if __name__ == "__main__":
    unittest.main()
