"""Classifier-training contract for the supervised DBN example."""

import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


def build_model(classifier_type="logistic"):
    """Builds only the attributes ``_train_classifier`` reads."""
    model = SupervisedDBNClassification.__new__(SupervisedDBNClassification)
    model.verbose = False
    model.classifier_type = classifier_type
    model.clf_C = 1.0
    model.clf_iter = 50
    model.random_state = 0

    def transform(X):
        return np.ones((X.shape[0], 2), dtype=np.float32)

    model.transform = transform
    return model


def test_logistic_classifier_is_trained_on_transformed_features():
    """The fitted classifier sees ``transform(X)``, not the raw input."""
    model = build_model("logistic")
    X = np.zeros((4, 3), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    model._train_classifier(X, y)
    assert isinstance(model.classifier, LogisticRegression)
    assert model.classifier.n_features_in_ == 2


def test_svm_classifier_enables_probability_estimates():
    """The SVM branch builds an SVC with probability support."""
    model = build_model("svm")
    model._train_classifier(np.zeros((4, 3), dtype=np.float32), np.array([0, 1, 0, 1]))
    assert isinstance(model.classifier, SVC)
    assert model.classifier.probability is True


def test_unknown_classifier_type_is_rejected():
    """Only the three documented families are supported."""
    model = build_model("bogus")
    with pytest.raises(ValueError, match="Unsupported classifier type"):
        model._train_classifier(
            np.zeros((4, 3), dtype=np.float32), np.array([0, 1, 0, 1])
        )
