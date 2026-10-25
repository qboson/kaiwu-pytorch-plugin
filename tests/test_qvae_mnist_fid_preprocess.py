"""FID image-preprocessing contract for the QVAE-MNIST helpers."""

import importlib.util
import os

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_fid_preprocess",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py"),
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def test_flat_mnist_images_become_three_channel_fid_inputs():
    """(N, 784) inputs are reshaped, triplicated and resized to 299x299."""
    preprocessor = helpers.FIDImagePreprocessor()
    images = torch.rand(2, 784)
    processed = preprocessor(images)
    assert processed.shape == (2, 3, 299, 299)
    assert processed.dtype == images.dtype
    for channel in range(1, 3):
        assert torch.equal(processed[:, channel], processed[:, 0])


def test_preprocessing_is_float_and_keeps_the_value_range():
    """Resizing does not change the [0, 1] value domain."""
    preprocessor = helpers.FIDImagePreprocessor()
    processed = preprocessor(torch.full((1, 784), 0.5))
    assert processed.min().item() >= 0.0
    assert processed.max().item() <= 1.0
