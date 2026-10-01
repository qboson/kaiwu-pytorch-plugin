"""Regression tests for AbstractBoltzmannMachine.to() contract compliance."""
import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import RestrictedBoltzmannMachine, BoltzmannMachine


class TestToContractRBM(unittest.TestCase):
    """Test AbstractBoltzmannMachine.to() contract compliance for RBM."""

    def setUp(self):
        self.model = RestrictedBoltzmannMachine(2, 3, device=torch.device("cpu"))

    def _assert_device_invariants(self):
        """Verify self.device is valid and matches actual parameter state."""
        param = next(self.model.parameters())
        self.assertIsInstance(self.model.device, torch.device)
        self.assertEqual(self.model.device, param.device)

    def test_device_only(self):
        result = self.model.to(torch.device("cpu"))
        self.assertIs(result, self.model)
        param = next(self.model.parameters())
        self.assertEqual(param.device.type, "cpu")
        self.assertEqual(param.dtype, torch.float32)
        self._assert_device_invariants()

        inp = torch.ones(1, self.model.num_nodes, dtype=param.dtype, device=param.device)
        energy = self.model(inp)
        self.assertEqual(energy.dtype, param.dtype)
        self.assertIsNotNone(self.model.get_ising_matrix())

    def test_positional_dtype(self):
        result = self.model.to(torch.float64)
        self.assertIs(result, self.model)
        param = next(self.model.parameters())
        self.assertEqual(param.dtype, torch.float64)
        self.assertEqual(param.device.type, "cpu")
        self._assert_device_invariants()

        inp = torch.ones(1, self.model.num_nodes, dtype=torch.float64)
        energy = self.model(inp)
        self.assertEqual(energy.dtype, torch.float64)
        self.assertIsNotNone(self.model.get_ising_matrix())

    def test_keyword_dtype(self):
        result = self.model.to(dtype=torch.float64)
        self.assertIs(result, self.model)
        param = next(self.model.parameters())
        self.assertEqual(param.dtype, torch.float64)
        self.assertEqual(param.device.type, "cpu")
        self._assert_device_invariants()

        inp = torch.ones(1, self.model.num_nodes, dtype=torch.float64)
        energy = self.model(inp)
        self.assertEqual(energy.dtype, torch.float64)

    def test_device_and_dtype(self):
        result = self.model.to(torch.device("cpu"), dtype=torch.float64)
        self.assertIs(result, self.model)
        param = next(self.model.parameters())
        self.assertEqual(param.device.type, "cpu")
        self.assertEqual(param.dtype, torch.float64)
        self._assert_device_invariants()

        inp = torch.ones(1, self.model.num_nodes, dtype=torch.float64)
        energy = self.model(inp)
        self.assertEqual(energy.dtype, torch.float64)
        self.assertIsNotNone(self.model.get_ising_matrix())

    def test_reference_tensor(self):
        reference = torch.zeros(1, dtype=torch.float64, device="cpu")
        result = self.model.to(reference)
        self.assertIs(result, self.model)
        param = next(self.model.parameters())
        self.assertEqual(param.device, reference.device)
        self.assertEqual(param.dtype, reference.dtype)
        self._assert_device_invariants()
        self.assertEqual(self.model.device, reference.device)

        inp = torch.ones(1, self.model.num_nodes, dtype=torch.float64)
        energy = self.model(inp)
        self.assertEqual(energy.dtype, torch.float64)

    def test_non_blocking_forwarding(self):
        original = torch.nn.Module.to
        captured = []

        def spy(self, *args, **kwargs):
            captured.append({"args": args, "kwargs": kwargs})
            return original(self, *args, **kwargs)

        try:
            torch.nn.Module.to = spy
            self.model.to(torch.device("cpu"), non_blocking=True)
            self.assertEqual(len(captured), 1)
            self.assertTrue(captured[0]["kwargs"].get("non_blocking") is True)
        finally:
            torch.nn.Module.to = original

    def test_memory_format_passthrough(self):
        result = self.model.to(memory_format=torch.channels_last)
        self.assertIs(result, self.model)
        self._assert_device_invariants()

    def test_exception_atomicity(self):
        device_before = self.model.device
        param_device_before = next(self.model.parameters()).device
        param_dtype_before = next(self.model.parameters()).dtype

        with self.assertRaises((TypeError, RuntimeError)):
            self.model.to(dtype=torch.int64)

        self.assertEqual(self.model.device, device_before)
        self.assertIsInstance(self.model.device, torch.device)
        self.assertEqual(next(self.model.parameters()).device, param_device_before)
        self.assertEqual(next(self.model.parameters()).dtype, param_dtype_before)


class TestToContractBM(unittest.TestCase):
    """Test AbstractBoltzmannMachine.to() contract compliance for BM."""

    def setUp(self):
        self.model = BoltzmannMachine(5, device=torch.device("cpu"))

    def _assert_device_invariants(self):
        param = next(self.model.parameters())
        self.assertIsInstance(self.model.device, torch.device)
        self.assertEqual(self.model.device, param.device)

    def test_device_only(self):
        result = self.model.to(torch.device("cpu"))
        self.assertIs(result, self.model)
        param = next(self.model.parameters())
        self.assertEqual(param.device.type, "cpu")
        self.assertEqual(param.dtype, torch.float32)
        self._assert_device_invariants()
        self.assertIsNotNone(self.model.get_ising_matrix())

    def test_positional_dtype(self):
        result = self.model.to(torch.float64)
        self.assertIs(result, self.model)
        param = next(self.model.parameters())
        self.assertEqual(param.dtype, torch.float64)
        self.assertEqual(param.device.type, "cpu")
        self._assert_device_invariants()

        inp = torch.ones(1, self.model.num_nodes, dtype=torch.float64)
        energy = self.model(inp)
        self.assertEqual(energy.dtype, torch.float64)
        self.assertIsNotNone(self.model.get_ising_matrix())

    def test_keyword_dtype(self):
        result = self.model.to(dtype=torch.float64)
        self.assertIs(result, self.model)
        param = next(self.model.parameters())
        self.assertEqual(param.dtype, torch.float64)
        self.assertEqual(param.device.type, "cpu")
        self._assert_device_invariants()

    def test_device_and_dtype(self):
        result = self.model.to(torch.device("cpu"), dtype=torch.float64)
        self.assertIs(result, self.model)
        param = next(self.model.parameters())
        self.assertEqual(param.device.type, "cpu")
        self.assertEqual(param.dtype, torch.float64)
        self._assert_device_invariants()
        self.assertIsNotNone(self.model.get_ising_matrix())

    def test_reference_tensor(self):
        reference = torch.zeros(1, dtype=torch.float64, device="cpu")
        result = self.model.to(reference)
        self.assertIs(result, self.model)
        param = next(self.model.parameters())
        self.assertEqual(param.device, reference.device)
        self.assertEqual(param.dtype, reference.dtype)
        self._assert_device_invariants()

    def test_non_blocking_forwarding(self):
        original = torch.nn.Module.to
        captured = []

        def spy(self, *args, **kwargs):
            captured.append({"args": args, "kwargs": kwargs})
            return original(self, *args, **kwargs)

        try:
            torch.nn.Module.to = spy
            self.model.to(torch.device("cpu"), non_blocking=True)
            self.assertEqual(len(captured), 1)
            self.assertTrue(captured[0]["kwargs"].get("non_blocking") is True)
        finally:
            torch.nn.Module.to = original

    def test_memory_format_passthrough(self):
        result = self.model.to(memory_format=torch.channels_last)
        self.assertIs(result, self.model)
        self._assert_device_invariants()

    def test_exception_atomicity(self):
        device_before = self.model.device
        param_device_before = next(self.model.parameters()).device
        param_dtype_before = next(self.model.parameters()).dtype

        with self.assertRaises((TypeError, RuntimeError)):
            self.model.to(dtype=torch.int64)

        self.assertEqual(self.model.device, device_before)
        self.assertIsInstance(self.model.device, torch.device)
        self.assertEqual(next(self.model.parameters()).device, param_device_before)
        self.assertEqual(next(self.model.parameters()).dtype, param_dtype_before)


if __name__ == "__main__":
    unittest.main()
