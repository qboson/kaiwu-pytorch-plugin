"""Real-image collection contract for the QVAE-MNIST helpers."""

import importlib.util
import os

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_real", os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py")
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def batches(sizes):
    """Builds a list of (images, labels) batches with increasing values."""
    result = []
    value = 0.0
    for size in sizes:
        images = torch.full((size, 1, 2, 2), value)
        result.append((images, torch.zeros(size, dtype=torch.long)))
        value += 1.0
    return result


def test_collection_stops_at_the_requested_count():
    """Whole batches are concatenated and then truncated to ``n_images``."""
    images = helpers.get_real_images(batches([4, 4, 4]), n_images=6)
    assert images.shape == (6, 1, 2, 2)
    assert images[0, 0, 0, 0].item() == 0.0
    assert images[5, 0, 0, 0].item() == 1.0


def test_short_dataloader_returns_every_image():
    """A loader with fewer images than requested is returned in full."""
    images = helpers.get_real_images(batches([2, 3]), n_images=100)
    assert images.shape == (5, 1, 2, 2)
