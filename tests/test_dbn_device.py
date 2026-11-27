import os
import sys
import unittest

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

import numpy as np
import torch

from kaiwu.torch_plugin import UnsupervisedDBN


class TestDBNDevice(unittest.TestCase):
    """Device selection and propagation for UnsupervisedDBN."""

    def test_default_device(self):
        """Without an argument the DBN picks CUDA when available, else CPU."""
        dbn = UnsupervisedDBN([4, 2])
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(dbn.device, expected)

    def test_explicit_device_object(self):
        """A torch.device is honored for the RBM layers."""
        dbn = UnsupervisedDBN([4, 2], device=torch.device("cpu"))
        self.assertEqual(dbn.device, torch.device("cpu"))
        dbn.create_rbm_layer(3)
        for rbm in dbn.rbm_layers:
            self.assertEqual(rbm.device, torch.device("cpu"))

    def test_explicit_device_string(self):
        """A device string is accepted and normalized."""
        dbn = UnsupervisedDBN([4, 2], device="cpu")
        self.assertEqual(dbn.device, torch.device("cpu"))

    def test_create_rbm_layer_uses_device(self):
        """RBM layers land on the configured device."""
        dbn = UnsupervisedDBN([4, 2], device="cpu")
        dbn.create_rbm_layer(3)
        self.assertEqual(len(dbn.rbm_layers), 2)
        self.assertTrue(all(r.device == torch.device("cpu") for r in dbn.rbm_layers))

    def test_forward_runs_on_configured_device(self):
        """A full forward pass works on the configured device."""
        dbn = UnsupervisedDBN([4, 2], device="cpu")
        dbn.create_rbm_layer(3)
        dbn.mark_as_trained()
        out = dbn.forward(np.random.rand(5, 3).astype(np.float32))
        self.assertEqual(out.shape, (5, 2))

    def test_to_keeps_device_in_sync(self):
        """to() moves parameters and updates self.device together."""
        dbn = UnsupervisedDBN([4, 2], device="cpu")
        dbn.create_rbm_layer(3)
        dbn.mark_as_trained()
        moved = dbn.to("cpu")
        self.assertIs(moved, dbn)
        self.assertEqual(dbn.device, torch.device("cpu"))
        self.assertTrue(all(r.device == torch.device("cpu") for r in dbn.rbm_layers))
        out = dbn.forward(np.random.rand(5, 3).astype(np.float32))
        self.assertEqual(out.shape, (5, 2))


class TestDBNStructureValidation(unittest.TestCase):
    """Layer-structure validation for UnsupervisedDBN."""

    def test_default_structure(self):
        """The default structure is still [100, 100]."""
        dbn = UnsupervisedDBN()
        self.assertEqual(dbn.hidden_layers_structure, [100, 100])

    def test_tuple_structure_accepted(self):
        """Tuples are accepted and stored as lists."""
        dbn = UnsupervisedDBN((4, 2))
        self.assertEqual(dbn.hidden_layers_structure, [4, 2])

    def test_non_sequence_rejected(self):
        """A bare int is rejected with a clear message."""
        with self.assertRaises(ValueError) as ctx:
            UnsupervisedDBN(4)
        self.assertIn("hidden_layers_structure", str(ctx.exception))

    def test_bad_sizes_rejected(self):
        """Zero, negative, fractional, and non-numeric sizes are rejected."""
        for bad in ([0, 2], [-1, 2], [2.5], [None], ["abc"]):
            with self.subTest(structure=bad):
                with self.assertRaises(ValueError):
                    UnsupervisedDBN(bad)

    def test_numpy_int_sizes_accepted(self):
        """numpy integers from array shapes are valid layer sizes."""
        dbn = UnsupervisedDBN([np.int64(4), np.int64(2)])
        self.assertEqual(dbn.hidden_layers_structure, [4, 2])


if __name__ == "__main__":
    unittest.main()
