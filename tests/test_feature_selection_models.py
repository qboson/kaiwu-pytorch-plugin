"""Constructor dimension and valid-forward checks for the example models."""

from enum import IntEnum
import unittest

import torch

from example.feature_selection.feature_selection_models import (
    SimpleLSTM,
    SimpleRNN,
    TinyCNN,
)


class Dimension(IntEnum):
    """A genuine positive integer subclass accepted by the model API."""

    THREE = 3


class FeatureSelectionModelTests(unittest.TestCase):
    """Use real constructors and CPU forwards without feature-selection training."""

    def check_invalid_dimensions(self, model, parameter):
        for value in (0, -1, True, False, 2.5, 2.0):
            with self.subTest(kind="invalid", model=model.__name__, value=repr(value)):
                try:
                    model(value)
                except Exception as exc:
                    self.assertIsInstance(exc, ValueError)
                    self.assertIn(parameter, str(exc))
                else:
                    self.fail("Constructor accepted an invalid dimension")

    def test_invalid_cnn_dimensions(self):
        self.check_invalid_dimensions(TinyCNN, "image_size")

    def test_invalid_recurrent_dimensions(self):
        for model in (SimpleRNN, SimpleLSTM):
            self.check_invalid_dimensions(model, "feature_dim")

    def test_valid_dimensions_and_forwards(self):
        for dimension in (1, 3, Dimension.THREE):
            width = int(dimension)
            for model_type in (TinyCNN, SimpleRNN, SimpleLSTM):
                with self.subTest(model=model_type.__name__, dimension=repr(dimension)):
                    torch.manual_seed(1178)
                    model = model_type(dimension).eval()
                    state = model.state_dict()
                    if model_type is TinyCNN:
                        inputs = torch.arange(2 * width * width, dtype=torch.float32)
                        inputs = inputs.reshape(2, width * width)
                        self.assertEqual(tuple(state["net.0.weight"].shape), (8, 1, 3, 3))
                        self.assertEqual(tuple(state["net.3.weight"].shape), (2, 8 * width * width))
                        self.assertEqual(model.image_size, width)
                    else:
                        inputs = torch.arange(2 * 3 * width, dtype=torch.float32)
                        inputs = inputs.reshape(2, 3, width)
                        prefix, gates = ("rnn", 1) if model_type is SimpleRNN else ("lstm", 4)
                        self.assertEqual(
                            tuple(state[f"{prefix}.weight_ih_l0"].shape), (8 * gates, width)
                        )
                        self.assertEqual(
                            tuple(state[f"{prefix}.weight_hh_l0"].shape), (8 * gates, 8)
                        )
                        self.assertEqual(tuple(state["head.weight"].shape), (2, 8))
                    with torch.no_grad():
                        outputs = model(inputs)
                    self.assertEqual(tuple(outputs.shape), (2, 2))
                    self.assertTrue(torch.isfinite(outputs).all().item())
                    self.assertEqual(outputs.dtype, torch.float32)
                    self.assertEqual(outputs.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
