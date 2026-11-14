"""Deterministic sklearn Digits splits and their PyTorch ``Dataset`` wrapper.

The Digits set ships inside scikit-learn: 1,797 grayscale 8x8 images of the
digits 0 to 9, with pixel values 0 to 16. Nothing is downloaded, so an
experiment runs on a machine with no network access.

The train/test split is stratified and fixed by ``split_seed``, which is
recorded in ``config.json`` and is independent of the experiment seeds: every
experiment seed sees the same images, so seeds differ only in initialization,
shuffling and annealing. An optional validation split is carved out of the
training part, never out of the test part.

Pixels are divided by 16 to lie in ``[0, 1]`` and flattened in C order, giving
64 inputs. There is no augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

#: Number of input pixels, 8 x 8.
N_INPUTS = 64
#: Number of digit classes.
N_CLASSES = 10
#: Largest raw pixel value in the Digits set.
PIXEL_MAX = 16.0
#: Default fraction of all images held out for testing.
TEST_FRACTION = 0.2
#: Default seed of the stratified split; independent of the experiment seeds.
SPLIT_SEED = 0


@dataclass(frozen=True)
class DigitsSplit:
    """One split of the Digits set.

    Attributes:
        images: Flattened ``float64`` pixels in ``[0, 1]``, one row per sample.
        labels: Integer class labels aligned with ``images``.
        indices: Positions of these samples in ``sklearn.datasets.load_digits``.
    """

    images: np.ndarray
    labels: np.ndarray
    indices: list[int]

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            ``len(labels)``.
        """
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class DigitsSplits:
    """Train, test and optional validation splits drawn from one partition.

    Attributes:
        train: Training split.
        test: Test split, never used for training or model selection.
        validation: Validation split taken from the training part, or ``None``.
        split_seed: Seed that fixed the partition.
        test_fraction: Fraction of all images in :attr:`test`.
        validation_fraction: Fraction of the training part moved to
            :attr:`validation`; ``0`` when there is none.
    """

    train: DigitsSplit
    test: DigitsSplit
    validation: DigitsSplit | None
    split_seed: int
    test_fraction: float
    validation_fraction: float

    def to_json(self) -> dict[str, object]:
        """Describe the partition for ``config.json``.

        Returns:
            Mapping with the dataset name, seeds, fractions and split sizes.
        """
        return {
            "dataset": "sklearn_digits",
            "n_inputs": N_INPUTS,
            "pixel_scale": f"x / {PIXEL_MAX:g}",
            "split_seed": self.split_seed,
            "test_fraction": self.test_fraction,
            "validation_fraction": self.validation_fraction,
            "n_train": len(self.train),
            "n_test": len(self.test),
            "n_validation": 0 if self.validation is None else len(self.validation),
        }


def normalize_pixels(raw: np.ndarray) -> np.ndarray:
    """Scale raw Digits pixels from ``0..16`` to ``[0, 1]`` and flatten them.

    Args:
        raw: Pixel array of shape ``(n, 64)`` or ``(n, 8, 8)``.

    Returns:
        A ``float64`` array of shape ``(n, 64)``.
    """
    pixels = np.asarray(raw, dtype=np.float64)
    return (pixels / PIXEL_MAX).reshape(pixels.shape[0], -1)


def _subset(images: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> DigitsSplit:
    """Build a split from positions into the full Digits arrays.

    Args:
        images: All normalized images.
        labels: All labels.
        indices: Positions to take, kept in ascending order.

    Returns:
        The split holding those samples.
    """
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    return DigitsSplit(
        images=images[ordered],
        labels=labels[ordered].astype(np.int64),
        indices=[int(position) for position in ordered],
    )


def load_digits_splits(
    *,
    split_seed: int = SPLIT_SEED,
    test_fraction: float = TEST_FRACTION,
    validation_fraction: float = 0.0,
) -> DigitsSplits:
    """Load stratified, disjoint train, test and optional validation splits.

    The test split is drawn first from all images. When ``validation_fraction``
    is positive, that fraction of the remaining training images becomes the
    validation split, again stratified. Within each split, samples keep their
    original order, so the split is a pure function of its arguments.

    Args:
        split_seed: Seed of the stratified partition.
        test_fraction: Fraction of all images held out for testing.
        validation_fraction: Fraction of the training part held out for
            validation. ``0`` disables the validation split.

    Returns:
        The three splits and the parameters that produced them.

    Raises:
        ValueError: If a fraction lies outside ``(0, 1)`` (``[0, 1)`` for
            ``validation_fraction``).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must lie in (0, 1), got {test_fraction}")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must lie in [0, 1), got {validation_fraction}")

    pixels, targets = load_digits(return_X_y=True)
    images = normalize_pixels(pixels)
    labels = np.asarray(targets, dtype=np.int64)
    positions = np.arange(labels.shape[0])

    train_positions, test_positions = train_test_split(
        positions,
        test_size=test_fraction,
        random_state=int(split_seed),
        stratify=labels,
    )
    validation = None
    if validation_fraction > 0.0:
        train_positions, validation_positions = train_test_split(
            train_positions,
            test_size=validation_fraction,
            random_state=int(split_seed),
            stratify=labels[train_positions],
        )
        validation = _subset(images, labels, validation_positions)

    return DigitsSplits(
        train=_subset(images, labels, train_positions),
        test=_subset(images, labels, test_positions),
        validation=validation,
        split_seed=int(split_seed),
        test_fraction=float(test_fraction),
        validation_fraction=float(validation_fraction),
    )


class DigitsDataset(Dataset):
    """``torch.utils.data.Dataset`` view of a :class:`DigitsSplit`.

    Items are ``(image, label)`` pairs: a 1-D tensor of 64 pixels in ``[0, 1]``
    and a scalar ``int64`` label, so the dataset works with a stock
    ``DataLoader`` and its default collation.

    Attributes:
        images: Pixel tensor of shape ``(n, 64)``.
        labels: Label tensor of shape ``(n,)``.
    """

    def __init__(self, split: DigitsSplit, dtype: torch.dtype = torch.float32) -> None:
        """Wrap a split.

        Args:
            split: Split to expose.
            dtype: Floating dtype of the image tensors.
        """
        self.images = torch.as_tensor(split.images, dtype=dtype)
        self.labels = torch.as_tensor(split.labels, dtype=torch.int64)

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            The number of images.
        """
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one sample.

        Args:
            index: Sample position.

        Returns:
            The image row and its label.
        """
        return self.images[index], self.labels[index]
