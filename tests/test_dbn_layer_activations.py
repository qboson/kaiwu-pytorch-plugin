"""Layer-activation contract for the supervised DBN classifier."""

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


class FakeRBM:
    """Concatenates the input with a scaled copy as the hidden state."""

    def __init__(self, num_visible=2, scale=10.0):
        self.num_visible = num_visible
        self.scale = scale

    def get_hidden(self, x):
        return torch.cat([x, x * self.scale], dim=1)


def build_model(n_layers=2):
    """Builds only the attributes ``get_layer_activations`` reads."""
    model = SupervisedDBNClassification.__new__(SupervisedDBNClassification)
    model.unsupervised_dbn = type(
        "StubDBN",
        (),
        {
            "_n_layers": n_layers,
            "device": torch.device("cpu"),
            "get_rbm_layer": lambda self, index: FakeRBM(scale=10.0 ** (index + 1)),
        },
    )()
    return model


def test_all_layers_are_returned_as_maps():
    """``return_all_layers`` returns one hidden activation per layer."""
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    activations = build_model().get_layer_activations(X, return_all_layers=True)
    assert set(activations) == {0, 1}
    assert activations[0].tolist() == [[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]]
    assert activations[1].shape == (3, 2)


def test_default_returns_the_last_layer_and_index_selects_one():
    """Without ``return_all_layers`` the last or the requested layer is returned."""
    X = np.zeros((2, 2), dtype=np.float64)
    model = build_model()
    assert model.get_layer_activations(X).shape == (2, 2)
    first = model.get_layer_activations(X, layer_index=0)
    assert first.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_out_of_range_index_and_missing_layers_are_rejected():
    """A too-large index or an unfitted model raises ``ValueError``."""
    X = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="out of range"):
        build_model(n_layers=2).get_layer_activations(X, layer_index=5)
    with pytest.raises(ValueError, match="No RBM layers"):
        build_model(n_layers=0).get_layer_activations(X)
