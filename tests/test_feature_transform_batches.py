"""Exercise FeatureExtractor batching with the real MNIST BasicEncoder."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn


def _load_component(name, filename):
    path = (Path(__file__).resolve().parents[1]
            / "example" / "qvae_mnist" / "model" / filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXTRACTOR = _load_component("_feature_batch_contract", "feature_extractor.py")
_NETWORKS = _load_component("_mnist_encoder_batch_contract", "networks.py")


class _EncoderBoundary(nn.Module):
    """Test model with the real MNIST encoder and four outputs."""

    def __init__(self):
        super().__init__()
        # Linear initialization uses only CPU RNG and leaves caller state intact.
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            self.encoder = _NETWORKS.BasicEncoder(
                node_sequence=[(4, 5), (5, 3)], activation_fct=nn.Tanh()
            )
        with torch.no_grad():
            for index, parameter in enumerate(self.encoder.parameters()):
                values = torch.arange(parameter.numel(), dtype=parameter.dtype)
                parameter.copy_((values.reshape(parameter.shape) + index)
                                / (parameter.numel() + 5) - 0.25)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(17)
        self.calls = []
        self.zeta_outputs = []

    def forward(self, x):
        self.calls.append({"rows": len(x), "training": self.training,
                           "grad_enabled": torch.is_grad_enabled()})
        q = self.encoder(x)
        zeta = torch.bernoulli(torch.sigmoid(q), generator=self.generator)
        self.zeta_outputs.append(zeta.detach().clone())
        return x + 1, None, q, zeta


class TestFeatureTransformBatches(unittest.TestCase):
    def setUp(self):
        self.model = _EncoderBoundary()

    def test_default_numpy_batches_match_full_basic_encoder(self):
        values = np.arange(513 * 4, dtype=np.float32).reshape(513, 4) / 1000
        with torch.no_grad():
            expected = self.model.encoder(torch.from_numpy(values)).numpy()

        result = _EXTRACTOR.FeatureExtractor(self.model).transform(values)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (513, 3))
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual([call["rows"] for call in self.model.calls], [256, 256, 1])
        self.assertTrue(all(not call["training"] for call in self.model.calls))
        self.assertTrue(all(not call["grad_enabled"] for call in self.model.calls))

    def test_tensor_custom_batches_preserve_order_and_q(self):
        values = torch.arange(36, dtype=torch.float32).reshape(9, 4) / 20
        cases = ((1, [1] * 9), (4, [4, 4, 1]),
                 (20, [9]), (np.int64(3), [3, 3, 3]))
        for size, expected_sizes in cases:
            with self.subTest(batch_size=size):
                self.model = _EncoderBoundary()
                with torch.no_grad():
                    expected = self.model.encoder(values).numpy()
                result = _EXTRACTOR.FeatureExtractor(
                    self.model, feature_type="q", batch_size=size
                ).transform(values)
                self.assertIsInstance(result, np.ndarray)
                np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual([call["rows"] for call in self.model.calls], expected_sizes)

    def test_zeta_aggregates_actual_selected_outputs_for_both_input_types(self):
        values = np.arange(36, dtype=np.float32).reshape(9, 4) / 20
        for data in (values, torch.from_numpy(values)):
            with self.subTest(input_type=type(data).__name__):
                self.model = _EncoderBoundary()
                result = _EXTRACTOR.FeatureExtractor(
                    self.model, feature_type="zeta", batch_size=4
                ).transform(data)
                expected = torch.cat(self.model.zeta_outputs).numpy()
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, (9, 3))
                np.testing.assert_array_equal(result, expected)
                self.assertEqual([call["rows"] for call in self.model.calls], [4, 4, 1])

    def test_invalid_batch_sizes_are_rejected_at_construction(self):
        for size in (0, -1, 1.5, "4", None, True, np.bool_(True)):
            with self.subTest(batch_size=size):
                with self.assertRaisesRegex(ValueError, "batch_size.*positive integer"):
                    _EXTRACTOR.FeatureExtractor(self.model, batch_size=size)
        self.assertEqual(self.model.calls, [])

    def test_empty_numpy_and_tensor_inputs_raise_before_forward(self):
        for data in (np.empty((0, 4), dtype=np.float32), torch.empty(0, 4)):
            with self.subTest(input_type=type(data).__name__):
                with self.assertRaisesRegex(ValueError, "at least one sample"):
                    _EXTRACTOR.FeatureExtractor(self.model).transform(data)
        self.assertEqual(self.model.calls, [])


if __name__ == "__main__":
    unittest.main()
