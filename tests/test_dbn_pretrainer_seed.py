"""Regression test for ``DBNPretrainer(random_state=...)`` reproducibility.

``DBNPretrainer.fit`` created the RBM layers (whose weights are drawn with
``torch.randn``) *before* ``DBNTrainer.train`` applied the seed, so two runs
with the same ``random_state`` produced different models.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402

from dbn_trainer import DBNPretrainer  # noqa: E402


def _fit_once(x):
    pretrainer = DBNPretrainer(
        hidden_layers_structure=[6],
        n_epochs_rbm=0,  # initialization only: no sampler/licence needed
        random_state=42,
        verbose=False,
    )
    pretrainer.fit(x)
    return pretrainer.get_rbm_layer(0).quadratic_coef.detach().clone()


def test_same_random_state_reproduces_the_initialization():
    x = np.random.RandomState(0).rand(16, 12).astype(np.float32)

    first = _fit_once(x)
    second = _fit_once(x)

    torch.testing.assert_close(first, second)


def test_different_random_states_still_differ():
    x = np.random.RandomState(0).rand(16, 12).astype(np.float32)

    first = _fit_once(x)

    pretrainer = DBNPretrainer(
        hidden_layers_structure=[6],
        n_epochs_rbm=0,
        random_state=7,
        verbose=False,
    )
    pretrainer.fit(x)
    second = pretrainer.get_rbm_layer(0).quadratic_coef.detach()

    assert not torch.allclose(first, second)
