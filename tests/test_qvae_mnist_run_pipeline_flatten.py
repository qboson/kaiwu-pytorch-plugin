"""Dataloader-flattening contract for the QVAE-MNIST pipeline script."""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

from run_pipeline import flatten_dataloader  # noqa: E402


class ListLoader:
    """Minimal iterable yielding (images, labels) batches."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def test_images_are_flattened_per_sample_and_stacked():
    """Each image row is flattened to 16 features and batches are stacked."""
    batches = [
        (torch.arange(8, dtype=torch.float32).view(2, 1, 2, 2), torch.tensor([0, 1])),
        (torch.arange(8, 16, dtype=torch.float32).view(2, 1, 2, 2), torch.tensor([2, 3])),
    ]
    X, y = flatten_dataloader(ListLoader(batches))
    assert isinstance(X, np.ndarray)
    assert X.shape == (4, 4)
    assert X.dtype == np.float32
    assert y.tolist() == [0, 1, 2, 3]
    assert X[0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert X[1].tolist() == [4.0, 5.0, 6.0, 7.0]
    assert X[2].tolist() == [8.0, 9.0, 10.0, 11.0]
    assert X[3].tolist() == [12.0, 13.0, 14.0, 15.0]


def test_labels_are_concatenated_in_batch_order():
    """Labels keep the loader order across batches."""
    batches = [
        (torch.zeros(1, 1, 2, 2), torch.tensor([7])),
        (torch.ones(2, 1, 2, 2), torch.tensor([8, 9])),
    ]
    X, y = flatten_dataloader(ListLoader(batches))
    assert X.shape == (3, 4)
    assert y.tolist() == [7, 8, 9]
