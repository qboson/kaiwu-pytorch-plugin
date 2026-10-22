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


def test_zero_validation_accuracy_keeps_actual_trained_weights(tmp_path, monkeypatch):
    """A valid run with no validation improvement must not restore None."""
    from downstream import classifier as classifier_module

    estimator = MLPClassifier(
        input_dim=4, hidden_dims=[8], output_dim=2, epochs_mlp=2,
        device="cpu", save_path=str(tmp_path)
    )
    initial = {}
    create_model = estimator._create_model

    def capture_initial_model():
        model = create_model()
        initial.update({key: value.detach().clone() for key, value in model.state_dict().items()})
        return model

    monkeypatch.setattr(estimator, "_create_model", capture_initial_model)
    # Only control the validation outcome; forward/backward/Adam remain real.
    monkeypatch.setattr(estimator, "_eval_mlp_epoch", lambda **kwargs: (0.0, 1.0))
    monkeypatch.setattr(classifier_module, "plot_training_curves", lambda **kwargs: None)
    features, labels = _toy_data()

    assert estimator.fit(features, labels) is estimator
    assert any(not torch.equal(initial[key], value)
               for key, value in estimator.model.state_dict().items())
    assert estimator.predict(features).shape == labels.shape
    assert not (tmp_path / "best_mlp_classifier.pth").exists()
    assert all(torch.isfinite(p).all() for p in estimator.model.parameters())
