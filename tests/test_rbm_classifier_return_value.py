"""Regression test for the predictions returned by train_classifier."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn")

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/rbm_digits"))
)
import rbm_classifier  # noqa: E402  pylint: disable=wrong-import-position


class _FakeRBM:
    """RBM step that passes features through unchanged."""

    def __init__(self, **kwargs):
        pass

    def load_data(self, plot_img=False):
        x = np.zeros((6, 3), dtype=np.float32)
        y = np.array([0, 1, 0, 1, 0, 1])
        return x[:4], x[4:], y[:4], y[4:]

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x

    def predict(self, x):
        return np.full(x.shape[0], 1)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kwargs):
        return self


class _FakeLogisticRegression:
    """First instance (pipeline) predicts 1, second (baseline) predicts 0."""

    instances = 0

    def __init__(self, **kwargs):
        _FakeLogisticRegression.instances += 1
        self._label = 1 if _FakeLogisticRegression.instances == 1 else 0
        self.C = 1.0
        self.max_iter = 100

    def fit(self, x, y):
        return self

    def predict(self, x):
        return np.full(x.shape[0], self._label)

    def get_params(self, deep=True):
        return {"C": self.C, "max_iter": self.max_iter, "random_state": None}

    def set_params(self, **kwargs):
        return self


def test_train_classifier_returns_rbm_pipeline_predictions(monkeypatch):
    monkeypatch.setattr(rbm_classifier, "RBMRunner", _FakeRBM)
    monkeypatch.setattr(rbm_classifier, "LogisticRegression", _FakeLogisticRegression)
    _FakeLogisticRegression.instances = 0

    _, _, returned_predictions = rbm_classifier.train_classifier(n_iter=1)

    assert np.all(returned_predictions == 1), (
        "train_classifier returned the raw-pixel baseline predictions while the "
        "notebook labels them as RBM-feature predictions"
    )
