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


def _trained_dbn():
    """Build a two-layer DBN with fixed parameters and mark it trained."""
    torch.manual_seed(0)
    dbn = UnsupervisedDBN(hidden_layers_structure=[4, 2])
    dbn.create_rbm_layer(3)
    dbn.mark_as_trained()
    return dbn


class TestForwardWithTensors(unittest.TestCase):
    """UnsupervisedDBN.forward/transform must accept torch tensors."""

    def setUp(self):
        self.dbn = _trained_dbn()

    def test_float32_tensor(self):
        """A float32 tensor works like the numpy path."""
        data = torch.rand(5, 3)
        out = self.dbn.forward(data)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (5, 2))

    def test_float64_tensor(self):
        """A float64 tensor is converted instead of raising AttributeError."""
        data = torch.rand(5, 3, dtype=torch.float64)
        out = self.dbn.forward(data)
        self.assertEqual(out.shape, (5, 2))

    def test_tensor_with_grad(self):
        """A grad-tracked tensor is detached internally, not rejected."""
        data = torch.rand(5, 3, requires_grad=True)
        out = self.dbn.forward(data)
        self.assertEqual(out.shape, (5, 2))
        self.assertFalse(data.grad is not None and data.grad.abs().sum() > 0)

    def test_numpy_input_unchanged(self):
        """The historical numpy path keeps its exact behavior."""
        data = np.random.rand(5, 3).astype(np.float64)
        out = self.dbn.forward(data)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (5, 2))

    def test_list_input(self):
        """Plain lists are accepted as a side effect of the conversion."""
        out = self.dbn.forward([[0.1, 0.2, 0.3]] * 5)
        self.assertEqual(out.shape, (5, 2))

    def test_transform_matches_forward(self):
        """transform() delegates to forward() for tensors as for arrays."""
        data = torch.rand(5, 3)
        self.assertTrue(
            np.array_equal(self.dbn.transform(data), self.dbn.forward(data))
        )

    def test_tensor_and_numpy_agree(self):
        """The same data as tensor and as numpy produces identical output."""
        array = np.random.rand(6, 3)
        out_array = self.dbn.forward(array)
        out_tensor = self.dbn.forward(torch.tensor(array))
        self.assertTrue(np.allclose(out_array, out_tensor))


class TestReconstructWithTensor(unittest.TestCase):
    """reconstruct_with_rbm must accept torch tensors."""

    def setUp(self):
        self.dbn = _trained_dbn()

    def test_tensor_input(self):
        """A tensor input returns reconstructions and per-sample errors."""
        data = torch.rand(4, 3)
        recon, errors = UnsupervisedDBN.reconstruct_with_rbm(
            self.dbn.rbm_layers[0], data
        )
        self.assertEqual(recon.shape, (4, 3))
        self.assertEqual(errors.shape, (4,))

    def test_numpy_and_tensor_agree(self):
        """Tensor and numpy inputs reconstruct identically."""
        array = np.random.rand(4, 3)
        recon_a, err_a = UnsupervisedDBN.reconstruct_with_rbm(
            self.dbn.rbm_layers[0], array
        )
        recon_t, err_t = UnsupervisedDBN.reconstruct_with_rbm(
            self.dbn.rbm_layers[0], torch.tensor(array)
        )
        self.assertTrue(np.allclose(recon_a, recon_t))
        self.assertTrue(np.allclose(err_a, err_t))

    def test_reconstruct_method_with_tensor(self):
        """The instance method also accepts tensors."""
        recon, errors = self.dbn.reconstruct(torch.rand(4, 3))
        self.assertEqual(recon.shape, (4, 3))
        self.assertEqual(errors.shape, (4,))


if __name__ == "__main__":
    unittest.main()
