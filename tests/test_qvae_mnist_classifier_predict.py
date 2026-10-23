"""Prediction contract for the QVAE-MNIST downstream classifier."""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

from downstream.classifier import MLPClassifier  # noqa: E402


class FixedLogits(torch.nn.Module):
    """Returns one fixed logit matrix for any input."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, x):
        return self.logits[: x.shape[0]]


def build_classifier(logits):
    """Builds a classifier around the fixed-logit model."""
    classifier = MLPClassifier.__new__(MLPClassifier)
    classifier.model = FixedLogits(logits)
    classifier.device = torch.device("cpu")
    return classifier


def test_predict_returns_the_argmax_class_per_row():
    """The predicted labels are the row-wise argmax over the logits."""
    logits = torch.tensor([[0.1, 2.0, 0.3], [5.0, 0.0, 1.0], [0.0, 0.0, 7.0]])
    classifier = build_classifier(logits)
    predictions = classifier.predict(np.zeros((3, 4), dtype=np.float32))
    assert predictions.tolist() == [1, 0, 2]


def test_score_reports_the_fraction_of_correct_labels():
    """The score delegates to the predictions and computes accuracy."""
    logits = torch.tensor([[0.1, 2.0], [5.0, 0.0], [0.0, 7.0], [4.0, 1.0]])
    classifier = build_classifier(logits)
    X = np.zeros((4, 3), dtype=np.float32)
    assert classifier.score(X, np.array([1, 0, 0, 0])) == 0.75
