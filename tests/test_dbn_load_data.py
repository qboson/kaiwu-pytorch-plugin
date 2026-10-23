"""Dataset-expansion contract for the DBN digits example."""

import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from supervised_dbn_digits import load_data  # noqa: E402


def test_dataset_is_expanded_with_four_translations_per_digit():
    """Each bundled digit contributes the original plus four shifts."""
    X_train, X_test, y_train, y_test = load_data(plot_img=False)
    total = X_train.shape[0] + X_test.shape[0]
    assert total == 5 * 1797
    assert X_train.shape[1] == 64
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
    assert y_train.dtype.kind == "i"


def test_features_are_flattened_and_scaled_to_the_unit_interval():
    """Min-max scaling keeps every pixel in [0, 1]."""
    X_train, X_test, _y_train, _y_test = load_data(plot_img=False)
    for split in (X_train, X_test):
        assert np.isfinite(split).all()
        assert split.min() >= -1e-12
        assert split.max() <= 1.0 + 1e-12


def test_split_is_deterministic_for_the_fixed_seed():
    """Repeated calls produce identical train/test tensors."""
    first = load_data(plot_img=False)
    second = load_data(plot_img=False)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
