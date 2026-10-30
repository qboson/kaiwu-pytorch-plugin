# -*- coding: utf-8 -*-
"""FID batching must accept the documented list/tuple input form."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

QVAE_MNIST_ROOT = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
if str(QVAE_MNIST_ROOT) not in sys.path:
    sys.path.insert(0, str(QVAE_MNIST_ROOT))

import utils.helpers as helpers  # noqa: E402  pylint: disable=wrong-import-position


class _RecordingFIDCalculator:
    """Record update batches without loading Inception weights."""

    instances: list["_RecordingFIDCalculator"] = []

    def __init__(self, device):
        self.device = device
        self.updates: list[tuple[tuple[int, ...], bool]] = []
        _RecordingFIDCalculator.instances.append(self)

    def update_batch(self, images: torch.Tensor, real: bool) -> None:
        self.updates.append((tuple(images.shape), real))

    def compute(self) -> float:
        return 1.25


@pytest.fixture(autouse=True)
def _patch_fid(monkeypatch):
    _RecordingFIDCalculator.instances.clear()
    monkeypatch.setattr(helpers, "FIDCalculator", _RecordingFIDCalculator)
    yield
    _RecordingFIDCalculator.instances.clear()


def _row_counts() -> tuple[int, int]:
    calculator = _RecordingFIDCalculator.instances[-1]
    real_rows = sum(shape[0] for shape, is_real in calculator.updates if is_real)
    fake_rows = sum(shape[0] for shape, is_real in calculator.updates if not is_real)
    return real_rows, fake_rows


def test_list_of_tensors_with_different_batch_sizes() -> None:
    """Batched tensors accumulate instead of failing tensor construction."""
    fake_imgs = [torch.rand(2, 784), torch.rand(3, 784)]
    real_imgs = [torch.rand(4, 784), torch.rand(1, 784)]

    score = helpers.compute_fid_in_batches(
        fake_imgs,
        real_imgs,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert score == 1.25
    assert _row_counts() == (5, 5)


def test_tuple_of_numpy_batches_with_different_shapes() -> None:
    """The same list/tuple contract holds for NumPy batch arrays."""
    fake_imgs = (np.zeros((2, 784), dtype=np.float32), np.ones((3, 784), dtype=np.float32))
    real_imgs = (np.zeros((4, 784), dtype=np.float32), np.ones((1, 784), dtype=np.float32))

    helpers.compute_fid_in_batches(
        fake_imgs,
        real_imgs,
        device=torch.device("cpu"),
        batch_size=3,
    )

    assert _row_counts() == (5, 5)


def test_single_tensor_path_is_unchanged() -> None:
    """A single tensor keeps its existing row count and batching."""
    helpers.compute_fid_in_batches(
        torch.rand(5, 784),
        torch.rand(6, 784),
        device=torch.device("cpu"),
        batch_size=4,
    )

    assert _row_counts() == (6, 5)
