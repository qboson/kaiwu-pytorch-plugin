"""Portable ModelTuner checkpoint loading without pretrained downloads."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn


def load_model_tuner():
    """Load the real helper without importing the full MNIST training stack."""
    example = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
    spec = importlib.util.spec_from_file_location(
        "mnist_checkpoint_model_tuner", example / "trainer" / "model_tuner.py"
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules), mock.patch.object(
        sys, "path", [str(example), *sys.path]
    ):
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                del sys.modules[name]
        spec.loader.exec_module(module)
    return module.ModelTuner


ModelTuner = load_model_tuner()


class TestMNISTCheckpointDevice(unittest.TestCase):
    """Restore real state dictionaries onto the registered model's device."""

    def test_load_restores_weights_and_requested_mode(self):
        """Keep CPU inference and the existing set_eval option unchanged."""
        for set_eval in (True, False):
            with self.subTest(set_eval=set_eval), tempfile.TemporaryDirectory() as directory:
                source = nn.Linear(2, 1, dtype=torch.float64)
                with torch.no_grad():
                    source.weight.copy_(torch.tensor([[2.0, -3.0]]))
                    source.bias.fill_(0.5)
                destination = nn.Linear(2, 1, dtype=torch.float64)
                tuner = ModelTuner()
                tuner.register_model(destination)
                tuner.infile = str(Path(directory) / "model.pt")
                torch.save(source.state_dict(), tuner.infile)

                tuner.load_model(set_eval=set_eval)

                self.assertEqual(destination.training, not set_eval)
                self.assertEqual(destination.weight.device, torch.device("cpu"))
                actual = destination(torch.tensor([[4.0, 1.0]], dtype=torch.float64))
                torch.testing.assert_close(
                    actual, torch.tensor([[5.5]], dtype=torch.float64)
                )

    def test_cuda_checkpoint_loads_on_cpu(self):
        """A checkpoint's saved CUDA location must not override a CPU model."""
        with tempfile.TemporaryDirectory() as directory:
            source = nn.Linear(2, 1, dtype=torch.float64)
            with torch.no_grad():
                source.weight.copy_(torch.tensor([[2.0, -3.0]]))
                source.bias.fill_(0.5)
            tuner = ModelTuner()
            destination = nn.Linear(2, 1, dtype=torch.float64)
            tuner.register_model(destination)
            tuner.infile = str(Path(directory) / "cuda_model.pt")
            if torch.cuda.is_available():
                torch.save(source.cuda().state_dict(), tuner.infile)
            else:
                # Use Torch's serializer to produce the same storage-location
                # metadata as a CUDA checkpoint on CPU-only CI runners.
                with mock.patch(
                    "torch.serialization.location_tag", return_value="cuda:0"
                ):
                    torch.save(source.state_dict(), tuner.infile)
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    torch.load(tuner.infile)

            tuner.load_model()

            self.assertFalse(destination.training)
            self.assertEqual(destination.weight.device, torch.device("cpu"))
            actual = destination(torch.tensor([[4.0, 1.0]], dtype=torch.float64))
            torch.testing.assert_close(
                actual, torch.tensor([[5.5]], dtype=torch.float64)
            )


if __name__ == "__main__":
    unittest.main()
