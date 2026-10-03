"""Regression tests for MLPClassifier.fit training-epoch handling.

``MLPClassifier(epochs_mlp=0).fit(...)`` used to crash deep inside torch
with ``TypeError: Expected state_dict to be dict-like, got NoneType``
because the best-model tracking never ran and ``load_state_dict(None)``
was called unconditionally.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402  pylint: disable=wrong-import-position

pytest.importorskip("sklearn")
pytest.importorskip("tqdm")

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/qvae_mnist"))
)

import numpy as np  # noqa: E402  pylint: disable=wrong-import-position
import torch  # noqa: E402  pylint: disable=wrong-import-position

from downstream.classifier import MLPClassifier  # noqa: E402  pylint: disable=wrong-import-position


def _toy_data(num_samples=40, num_features=4):
    rng = np.random.default_rng(0)
    features = rng.standard_normal((num_samples, num_features)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int64)
    return features, labels


def test_fit_rejects_zero_training_epochs_with_clear_error():
    """Zero epochs must fail fast with a clear message, not a torch TypeError."""
    classifier = MLPClassifier(
        input_dim=4, hidden_dims=[8], output_dim=2, epochs_mlp=0, save_path=None
    )
    features, labels = _toy_data()
    with pytest.raises(ValueError, match="epochs_mlp must be a positive integer"):
        classifier.fit(features, labels)


def test_fit_rejects_negative_training_epochs():
    """Negative epochs fail with the same fast, clear error."""
    classifier = MLPClassifier(
        input_dim=4, hidden_dims=[8], output_dim=2, epochs_mlp=-3, save_path=None
    )
    features, labels = _toy_data()
    with pytest.raises(ValueError, match="epochs_mlp must be a positive integer"):
        classifier.fit(features, labels)


def test_fit_single_epoch_returns_fitted_model(tmp_path):
    """A one-epoch run trains, restores the best state, and predicts labels."""
    classifier = MLPClassifier(
        input_dim=4, hidden_dims=[8], output_dim=2, epochs_mlp=1, save_path=str(tmp_path)
    )
    features, labels = _toy_data()
    result = classifier.fit(features, labels)
    assert result is classifier
    assert classifier.model is not None
    assert (tmp_path / "best_mlp_classifier.pth").exists()
    predictions = classifier.predict(features)
    assert predictions.shape == (features.shape[0],)
    assert set(np.unique(predictions)) <= {0, 1}
    for parameter in classifier.model.parameters():
        assert torch.isfinite(parameter).all()
