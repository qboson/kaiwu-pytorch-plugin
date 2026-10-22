"""Regression test for fine-tuning a DBN without pretrained RBM layers.

``hidden_layers_structure=[]`` is a legal configuration (no hidden layers) and
works in classifier mode, but fine-tuning crashed while building its network:

    UnboundLocalError: local variable 'input_size' referenced before assignment

because ``input_size`` was only assigned inside ``if _n_layers > 0`` while the
output layer is created unconditionally.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


def _data():
    rng = np.random.RandomState(0)
    x = rng.rand(40, 12).astype(np.float32)
    y = np.array([0, 1, 2, 3] * 10)
    return x, y


def test_fine_tuning_without_hidden_layers():
    x, y = _data()
    model = SupervisedDBNClassification(
        hidden_layers_structure=[],
        fine_tuning=True,
        n_iter_backprop=2,
        verbose=False,
    )

    model.fit(x, y)  # must not raise

    assert model.predict(x).shape == (x.shape[0],)
    first_layer = model.fine_tune_network[0]
    assert first_layer.in_features == x.shape[1]


def test_classifier_mode_without_hidden_layers_still_works():
    x, y = _data()
    model = SupervisedDBNClassification(
        hidden_layers_structure=[],
        fine_tuning=False,
        verbose=False,
    )

    model.fit(x, y)

    assert model.predict(x).shape == (x.shape[0],)
