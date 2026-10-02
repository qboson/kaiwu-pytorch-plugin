"""Regression test for ``SupervisedDBNClassification.get_feature_importance``.

For a multiclass downstream classifier ``classifier.coef_`` has one row per
class.  The method returned ``np.abs(coef_[0])``, i.e. only the first class's
coefficients, while presenting them as the model's feature importance.  The
example classifies the 10-digit dataset, where that silently drops 9 of the
10 coefficient rows.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


def _fitted_classifier(n_classes=3, n_features=8, n_samples=60):
    rng = np.random.RandomState(0)
    x = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, n_samples)
    classifier = SupervisedDBNClassification(
        hidden_layers_structure=[4], fine_tuning=False
    )
    classifier.classifier = LogisticRegression(max_iter=200).fit(x, y)
    return classifier


def test_multiclass_importance_aggregates_all_classes():
    classifier = _fitted_classifier(n_classes=3)
    coef = np.abs(classifier.classifier.coef_)
    assert coef.shape[0] == 3

    importance = classifier.get_feature_importance()

    assert importance.shape == (coef.shape[1],)
    np.testing.assert_allclose(importance, coef.mean(axis=0))
    assert not np.allclose(importance, coef[0]), (
        "multiclass importance must not be just the first class's coefficients"
    )


def test_binary_importance_is_unchanged():
    classifier = _fitted_classifier(n_classes=2)
    coef = np.abs(classifier.classifier.coef_)
    assert coef.shape[0] == 1

    importance = classifier.get_feature_importance()

    np.testing.assert_allclose(importance, coef[0])


def test_importance_returns_none_without_coefficients():
    classifier = _fitted_classifier()
    classifier.classifier = type("NoCoef", (), {})()
    assert classifier.get_feature_importance() is None
