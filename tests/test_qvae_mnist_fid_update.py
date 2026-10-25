"""FID batch-update contract for the QVAE-MNIST helpers."""

import importlib.util
import os

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_fid_update",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py"),
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


class RecordingFID:
    """Records the tensors and flags handed to torchmetrics FID."""

    def __init__(self):
        self.updates = []

    def update(self, images, real):
        self.updates.append((images, real))


class RecordingPreprocessor:
    """Records inputs and returns them reshaped for the FID model."""

    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, images):
        self.calls.append(images)
        return self.output


def build_calculator(device="cpu"):
    """Builds an FIDCalculator around recorders."""
    calculator = helpers.FIDCalculator.__new__(helpers.FIDCalculator)
    calculator.device = torch.device(device)
    calculator.fid = RecordingFID()
    calculator.preprocessor = RecordingPreprocessor(torch.zeros(2, 3, 8, 8))
    return calculator


def test_images_are_scaled_to_uint8_before_preprocessing():
    """[0,1] floats become 0..255 uint8 and the real flag is forwarded."""
    calculator = build_calculator()
    images = torch.tensor([[0.0, 0.5, 1.0, 2.0] * 196])
    calculator.update_batch(images, real=True)
    sent = calculator.preprocessor.calls[0]
    assert sent.dtype == torch.uint8
    assert sent.min().item() == 0
    assert sent.max().item() == 255
    processed, real = calculator.fid.updates[0]
    assert real is True
    assert torch.equal(processed, calculator.preprocessor.output)


def test_fake_batches_are_forwarded_with_real_false():
    """The fake side of FID is forwarded unchanged."""
    calculator = build_calculator()
    calculator.update_batch(torch.zeros(2, 784), real=False)
    _processed, real = calculator.fid.updates[0]
    assert real is False
