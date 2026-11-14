"""Tests for the validation split carved out of the MNIST training file."""

# Test functions document intent in their summary.
# pylint: disable=missing-param-doc,missing-return-doc

from __future__ import annotations

import gzip
from pathlib import Path
import struct

import numpy as np

from examples.mnist_kaiwu.data import (
    MNIST_ARCHIVES,
    first_per_class_indices,
    load_validation_split,
    mnist_raw_dir,
)


def _write_fake_mnist(root: Path, labels: np.ndarray) -> None:
    """Write tiny IDX archives whose pixel value encodes the sample position."""
    raw = mnist_raw_dir(root)
    raw.mkdir(parents=True)
    pixels = np.zeros((len(labels), 28, 28), dtype=np.uint8)
    pixels[:, 0, 0] = np.arange(len(labels)) % 256
    for name in MNIST_ARCHIVES:
        with gzip.open(raw / name, "wb") as handle:
            if "labels" in name:
                header = struct.pack(">II", 2049, len(labels))
                handle.write(header + labels.astype(np.uint8).tobytes())
            else:
                handle.write(struct.pack(">IIII", 2051, len(labels), 28, 28) + pixels.tobytes())


def test_validation_split_is_disjoint_from_training_and_class_balanced(tmp_path: Path) -> None:
    """Validation takes images 100..109 of each class, none of the first 100."""
    labels = np.tile(np.arange(10), 120)
    _write_fake_mnist(tmp_path, labels)

    split = load_validation_split(tmp_path, 10)

    train = set(first_per_class_indices(labels, 100))
    assert len(split.indices) == 100
    assert not train & set(split.indices)
    assert np.bincount(split.labels, minlength=10).tolist() == [10] * 10
    assert split.images.shape == (100, 784)
