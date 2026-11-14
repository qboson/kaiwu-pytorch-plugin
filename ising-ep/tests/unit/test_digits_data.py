"""Tests for the deterministic sklearn Digits splits and their Dataset wrapper."""

# Test functions document intent in their summary.
# pylint: disable=missing-param-doc,missing-return-doc

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from examples.digits_kaiwu.data import (
    N_CLASSES,
    N_INPUTS,
    DigitsDataset,
    load_digits_splits,
    normalize_pixels,
)

N_DIGITS = 1797


def test_default_split_is_a_disjoint_80_20_partition_of_every_image() -> None:
    """Train and test together cover all 1,797 images exactly once."""
    splits = load_digits_splits()

    train, test = set(splits.train.indices), set(splits.test.indices)
    assert (len(train), len(test)) == (1437, 360)
    assert not train & test
    assert train | test == set(range(N_DIGITS))
    assert splits.validation is None


def test_test_split_is_stratified_by_class() -> None:
    """Each class contributes about a fifth of its images to the test split."""
    splits = load_digits_splits()

    all_counts = np.bincount(
        np.concatenate([splits.train.labels, splits.test.labels]), minlength=N_CLASSES
    )
    test_counts = np.bincount(splits.test.labels, minlength=N_CLASSES)
    assert np.all(np.abs(test_counts - 0.2 * all_counts) <= 1)


def test_split_is_a_pure_function_of_its_seed() -> None:
    """The same split seed rebuilds the same partition; another seed does not."""
    first = load_digits_splits(split_seed=0)
    again = load_digits_splits(split_seed=0)
    other = load_digits_splits(split_seed=1)

    assert first.test.indices == again.test.indices
    np.testing.assert_array_equal(first.train.images, again.train.images)
    assert first.test.indices != other.test.indices


def test_pixels_are_scaled_to_the_unit_interval_and_flattened() -> None:
    """Raw values 0..16 become 64 floats in [0, 1]."""
    splits = load_digits_splits()

    assert splits.train.images.shape == (1437, N_INPUTS)
    assert splits.train.images.dtype == np.float64
    assert splits.train.images.min() == 0.0
    assert splits.train.images.max() == 1.0
    np.testing.assert_array_equal(
        normalize_pixels(np.full((2, 8, 8), 16.0)), np.ones((2, N_INPUTS))
    )


def test_validation_comes_out_of_the_training_part_only() -> None:
    """Validation is disjoint from train and test, and the test split is unchanged."""
    plain = load_digits_splits()
    splits = load_digits_splits(validation_fraction=0.2)

    assert splits.validation is not None
    train, test = set(splits.train.indices), set(splits.test.indices)
    validation = set(splits.validation.indices)
    assert splits.test.indices == plain.test.indices
    assert not validation & train
    assert not validation & test
    assert validation | train == set(plain.train.indices)
    assert len(validation) == 288


@pytest.mark.parametrize(
    ("test_fraction", "validation_fraction"), [(0.0, 0.0), (1.0, 0.0), (0.2, -0.1), (0.2, 1.0)]
)
def test_fractions_outside_their_range_are_rejected(
    test_fraction: float, validation_fraction: float
) -> None:
    """A split that would be empty or cover everything is refused."""
    with pytest.raises(ValueError, match="fraction"):
        load_digits_splits(test_fraction=test_fraction, validation_fraction=validation_fraction)


def test_to_json_records_everything_needed_to_rebuild_the_split() -> None:
    """``config.json`` gets the dataset, seed, fractions and sizes."""
    payload = load_digits_splits(split_seed=3, validation_fraction=0.1).to_json()

    assert payload["dataset"] == "sklearn_digits"
    assert payload["split_seed"] == 3
    assert payload["validation_fraction"] == 0.1
    assert payload["n_train"] + payload["n_validation"] + payload["n_test"] == N_DIGITS


def test_dataset_items_are_pixel_rows_and_integer_labels() -> None:
    """``DigitsDataset`` behaves like any map-style Torch dataset."""
    split = load_digits_splits().test
    dataset = DigitsDataset(split)

    image, label = dataset[5]
    assert len(dataset) == len(split)
    assert image.shape == (N_INPUTS,)
    assert image.dtype == torch.float32
    assert label.dtype == torch.int64
    assert int(label) == int(split.labels[5])


def test_dataset_batches_through_a_stock_dataloader() -> None:
    """The default collation stacks images and labels into batch tensors."""
    loader = DataLoader(DigitsDataset(load_digits_splits().test), batch_size=32)

    images, labels = next(iter(loader))
    assert images.shape == (32, N_INPUTS)
    assert labels.shape == (32,)
    assert sum(batch[1].shape[0] for batch in loader) == 360
