"""Downstream-classifier delegation contract for the supervised DBN."""

import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


class RecordingClassifier:
    """Classifies the transformed features and records them."""

    def __init__(self):
        self.seen = []

    def predict(self, features):
        self.seen.append(("predict", features.copy()))
        return np.zeros(features.shape[0], dtype=int)

    def predict_proba(self, features):
        self.seen.append(("proba", features.copy()))
        return np.full((features.shape[0], 2), 0.5)


def build_model():
    """Builds a model whose transform returns a marker feature matrix."""
    model = SupervisedDBNClassification.__new__(SupervisedDBNClassification)
    model.classifier = RecordingClassifier()

    def transform(X):
        return X + 100.0

    model.transform = transform
    return model


def test_predict_uses_the_transformed_features():
    """Predictions are made on ``transform(X)`` rather than the raw input."""
    model = build_model()
    X = np.zeros((3, 2))
    labels = model._predict_with_classifier(X)
    assert labels.tolist() == [0, 0, 0]
    kind, features = model.classifier.seen[-1]
    assert kind == "predict"
    assert features.tolist() == [[100.0, 100.0]] * 3


def test_predict_proba_uses_the_transformed_features():
    """Probabilities are also computed on the transformed features."""
    model = build_model()
    X = np.ones((2, 2))
    proba = model._predict_proba_classifier(X)
    assert proba.shape == (2, 2)
    kind, features = model.classifier.seen[-1]
    assert kind == "proba"
    assert features.tolist() == [[101.0, 101.0]] * 2
