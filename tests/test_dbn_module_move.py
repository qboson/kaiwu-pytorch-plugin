"""Regression coverage for DBN device bookkeeping after module moves."""

import unittest

try:
    import numpy as np
    import torch
    from kaiwu.torch_plugin.dbn import UnsupervisedDBN
except ModuleNotFoundError:
    # Some local gates use a Python without PyTorch or the optional Kaiwu SDK.
    np = torch = UnsupervisedDBN = None


@unittest.skipIf(UnsupervisedDBN is None, "requires PyTorch and Kaiwu SDK")
class TestDBNModuleMove(unittest.TestCase):
    def test_inference_uses_independently_moved_layer(self):
        model = UnsupervisedDBN([3]).create_rbm_layer(2).mark_as_trained()
        model.rbm_layers[0].to("cpu")
        model.device = torch.device("meta")

        batch = np.zeros((2, 2), dtype=np.float32)
        self.assertEqual(model.forward(batch).shape, (2, 3))
        reconstructed, errors = model.reconstruct(batch)
        self.assertEqual(reconstructed.shape, (2, 2))
        self.assertEqual(errors.shape, (2,))

    def test_parent_apply_resynchronizes_devices(self):
        model = UnsupervisedDBN([3]).create_rbm_layer(2).mark_as_trained()
        model.device = torch.device("meta")

        model.to("cpu")

        self.assertEqual(model.device, torch.device("cpu"))
        self.assertEqual(model.rbm_layers[0].device, torch.device("cpu"))
        batch = np.zeros((2, 2), dtype=np.float32)
        self.assertEqual(model.forward(batch).shape, (2, 3))
        reconstructed, errors = model.reconstruct(batch)
        self.assertEqual(reconstructed.shape, (2, 2))
        self.assertEqual(errors.shape, (2,))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_cuda_to_cpu_inference(self):
        model = UnsupervisedDBN([3]).create_rbm_layer(2).mark_as_trained()
        self.assertEqual(model.rbm_layers[0].quadratic_coef.device.type, "cuda")

        model.to("cpu")

        batch = np.zeros((2, 2), dtype=np.float32)
        self.assertEqual(model.forward(batch).shape, (2, 3))
        self.assertEqual(model.reconstruct(batch)[0].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
