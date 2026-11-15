"""Contract tests for the Boltzmann machine base class."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin.abstract_boltzmann_machine import AbstractBoltzmannMachine


class TestAbstractBoltzmannMachine(unittest.TestCase):
    """The base class must refuse to compute what it cannot compute."""

    def setUp(self):
        self.bm = AbstractBoltzmannMachine(device=torch.device("cpu"))

    def test_forward_reports_the_missing_implementation(self):
        with self.assertRaises(NotImplementedError):
            self.bm(torch.ones(2, 3))

    def test_ising_matrix_reports_the_missing_implementation(self):
        with self.assertRaises(NotImplementedError):
            self.bm.get_ising_matrix()

    def test_default_device_is_a_torch_device(self):
        self.assertIsInstance(AbstractBoltzmannMachine().device, torch.device)


if __name__ == "__main__":
    unittest.main()
