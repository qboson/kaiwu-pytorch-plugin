"""FID's Inception input should retain the 8-bit brightness of MNIST images."""

import numpy as np
import pytest
import torch

for dependency in ("gif", "imageio", "pandas", "torchmetrics", "torchvision"):
    pytest.importorskip(dependency)

from example.qvae_mnist.utils import helpers


@pytest.mark.parametrize("as_tensor", [False, True])
def test_fid_receives_the_same_pixels_from_byte_and_normalized_float_inputs(monkeypatch, as_tensor):
    calls = []

    class RecordingFID:
        def __init__(self, feature):
            pass

        def to(self, device):
            return self

        def update(self, images, *, real):
            calls.append((real, images))

        def compute(self):
            return torch.tensor(0.0)

    monkeypatch.setattr(helpers, "FrechetInceptionDistance", RecordingFID)
    # No pretrained model or download. The actual TorchVision resize and RGB
    # preprocessing still run on both inputs, including the short final batch.
    pixels = np.broadcast_to(
        np.array([16, 128, 240], dtype=np.uint8)[:, None], (3, 784)
    ).copy()
    real = torch.as_tensor(pixels).to(torch.float32) / 255
    fake = torch.as_tensor(pixels) if as_tensor else pixels
    helpers.compute_fid_in_batches(fake, real, torch.device("cpu"), batch_size=2)

    assert [real for real, _ in calls] == [True, True, False, False]
    real_images = torch.cat([images for _, images in calls[:2]])
    fake_images = torch.cat([images for _, images in calls[2:]])
    torch.testing.assert_close(fake_images, real_images)
    torch.testing.assert_close(fake_images[:, 0, 0, 0], torch.tensor([16, 128, 240], dtype=torch.uint8))
