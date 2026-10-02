"""Regression test for ``n_iter_backprop=0`` in the supervised DBN.

``n_iter_backprop`` is a documented constructor argument (default 100).  With
0 the epoch loop never runs, so the post-loop summary read unbound variables
(``UnboundLocalError`` with ``verbose=True``) or, worse, the fine-tuning
network was returned untouched and randomly initialised (``verbose=False``),
silently producing meaningless predictions.  The configuration must fail fast.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402

from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


class _StubDBN:
    def __init__(self, rbm):
        self._rbm = rbm
        self.device = torch.device("cpu")
        self._n_layers = 1

    def get_rbm_layer(self, index):
        return self._rbm if index == 0 else None


def _classifier(n_iter_backprop, verbose):
    torch.manual_seed(0)
    rbm = RestrictedBoltzmannMachine(num_visible=6, num_hidden=4)
    classifier = SupervisedDBNClassification(
        hidden_layers_structure=[4],
        fine_tuning=True,
        n_iter_backprop=n_iter_backprop,
        verbose=verbose,
    )
    classifier.unsupervised_dbn = _StubDBN(rbm)
    classifier.classes_ = np.array([0, 1, 2])
    classifier._build_fine_tune_network()
    return classifier


@pytest.mark.parametrize("verbose", [True, False])
def test_zero_backprop_iterations_are_rejected(verbose):
    classifier = _classifier(n_iter_backprop=0, verbose=verbose)
    x = torch.rand(8, 6)
    y = torch.randint(0, 3, (8,))

    with pytest.raises(ValueError) as excinfo:
        classifier._train_fine_tune_network(x, y)

    assert "n_iter_backprop" in str(excinfo.value)


def test_positive_backprop_iterations_still_train():
    classifier = _classifier(n_iter_backprop=2, verbose=False)
    x = torch.rand(8, 6)
    y = torch.randint(0, 3, (8,))

    classifier._train_fine_tune_network(x, y)  # must not raise
