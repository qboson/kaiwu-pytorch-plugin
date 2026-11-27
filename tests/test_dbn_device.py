"""DBN inference follows its RBM parameters after module moves."""

import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from kaiwu.torch_plugin import UnsupervisedDBN


@unittest.skipIf(torch is None, "requires PyTorch")
class TestDBNDevice(unittest.TestCase):
    def test_inference_and_new_layers_after_move(self):
        dbn = UnsupervisedDBN([3]).create_rbm_layer(4).cpu()
        sample = np.zeros((1, 4), dtype=np.float32)

        self.assertEqual(dbn.mark_as_trained().forward(sample).shape, (1, 3))
        self.assertEqual(dbn.reconstruct(sample)[0].shape, (1, 4))
        self.assertEqual(dbn.reconstruct_with_rbm(dbn.rbm_layers[0], sample)[0].shape, (1, 4))

        dbn.create_rbm_layer(4)
        self.assertEqual(dbn.rbm_layers[0].quadratic_coef.device.type, "cpu")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_moves_between_cuda_and_cpu(self):
        dbn = UnsupervisedDBN([3]).create_rbm_layer(4)
        sample = np.zeros((1, 4), dtype=np.float32)

        for device in ("cpu", "cuda", "cpu"):
            dbn.to(device).mark_as_trained()
            self.assertEqual(dbn.forward(sample).shape, (1, 3))
            self.assertEqual(dbn.reconstruct(sample)[0].shape, (1, 4))

        dbn.to("cuda").create_rbm_layer(4)
        self.assertEqual(dbn.rbm_layers[0].quadratic_coef.device.type, "cuda")
        self.assertEqual(dbn.mark_as_trained().forward(sample).shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
