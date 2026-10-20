"""Regression tests for held-out data in the optional digits examples."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from sklearn.datasets import load_digits


@pytest.fixture(params=["rbm", "dbn"])
def digits_loader(request, monkeypatch):
    """Load the actual example entry points without training or sampling."""
    root = Path(__file__).resolve().parents[1]
    if request.param == "rbm":
        path = root / "example/rbm_digits/rbm_digits.py"
    else:
        path = root / "example/dbn_digits/supervised_dbn_digits.py"
        monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location(f"digits_example_{request.param}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if request.param == "rbm":
        # Loading data does not need an RBM or a remote sampler.
        runner = module.RBMRunner.__new__(module.RBMRunner)
        return module, runner.load_data
    return module, module.load_data


@pytest.mark.parametrize("data_source", ["synthetic", "bundled"])
def test_source_images_are_disjoint_across_splits(digits_loader, monkeypatch, data_source):
    """A translated sibling of a training image must never become a test row."""
    if data_source == "bundled":
        images = load_digits().images
    else:
        images = np.arange(20 * 64, dtype=float).reshape(20, 8, 8)
    # Unique labels track source-image lineage, independent of pixel transforms.
    source_ids = np.arange(len(images))
    module, load_data = digits_loader
    monkeypatch.setattr(
        module, "load_digits", lambda: SimpleNamespace(images=images, target=source_ids)
    )

    x_train, x_test, y_train, y_test = load_data()

    assert set(y_train).isdisjoint(y_test)
    np.testing.assert_array_equal(np.union1d(y_train, y_test), source_ids)
    train_sources, train_counts = np.unique(y_train, return_counts=True)
    test_sources, test_counts = np.unique(y_test, return_counts=True)
    np.testing.assert_array_equal(train_counts, np.full(len(train_sources), 5))
    np.testing.assert_array_equal(test_counts, np.ones(len(test_sources), dtype=int))
    assert len(test_sources) == int(np.ceil(0.2 * len(images)))
    assert x_train.shape == (5 * len(train_sources), 64)
    assert x_test.shape == (len(test_sources), 64)


def test_original_test_rows_use_only_training_scale(digits_loader, monkeypatch):
    """Check exact labels, shifts and scaling against an independent CPU fixture."""
    images = 1.0 + np.arange(20)[:, None, None] * 100.0
    images = images + np.arange(64).reshape(1, 8, 8)
    images[0] = 10000.0  # Source 0 belongs only to the fixed held-out split.
    source_ids = np.arange(len(images))
    module, load_data = digits_loader
    monkeypatch.setattr(
        module, "load_digits", lambda: SimpleNamespace(images=images, target=source_ids)
    )
    # The existing 20% split with random_state=42, before any augmentation.
    test_ids = np.array([0, 17, 15, 1])
    train_ids = np.array([8, 5, 11, 3, 18, 16, 13, 2, 9, 19, 4, 12, 7, 10, 14, 6])
    expected_images = []
    for source_id in train_ids:
        image = images[source_id]
        up, down, left, right = (np.zeros_like(image) for _ in range(4))
        up[:-1, :] = image[1:, :]
        down[1:, :] = image[:-1, :]
        left[:, :-1] = image[:, 1:]
        right[:, 1:] = image[:, :-1]
        expected_images.extend([image, up, down, left, right])
    expected_train = np.asarray(expected_images).reshape(-1, 64)
    minimum = expected_train.min(axis=0)
    span = expected_train.max(axis=0) - minimum
    expected_test = (images[test_ids].reshape(-1, 64) - minimum) / span
    expected_train = (expected_train - minimum) / span

    x_train, x_test, y_train, y_test = load_data()

    np.testing.assert_array_equal(y_train, np.repeat(train_ids, 5))
    np.testing.assert_array_equal(y_test, test_ids)
    np.testing.assert_allclose(x_train, expected_train, atol=1e-12)
    np.testing.assert_allclose(x_test, expected_test, atol=1e-12)
    assert x_test[0].min() > 1.0  # The held-out extreme did not enter scaler.fit.
