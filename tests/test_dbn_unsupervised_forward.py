"""Stacked-forward contract for the unsupervised DBN."""

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from kaiwu.torch_plugin.dbn import UnsupervisedDBN  # noqa: E402


class FakeRBM:
    """Concatenates the input with a scaled copy as the hidden state."""

    def __init__(self, num_visible, scale):
        self.num_visible = num_visible
        self.scale = scale

    def get_hidden(self, x):
        return torch.cat([x, x * self.scale], dim=1)


def build_model(layers=2, trained=True):
    """Builds only the attributes ``forward`` reads."""
    model = UnsupervisedDBN.__new__(UnsupervisedDBN)
    model.rbm_layers = [FakeRBM(2, 10.0 ** (index + 1)) for index in range(layers)]
    model._is_trained = trained
    model.device = torch.device("cpu")
    return model


def test_each_layer_consumes_only_the_hidden_half():
    """The output is the stacked hidden representation of the last layer."""
    model = build_model()
    data = np.arange(4, dtype=np.float32).reshape(2, 2)
    output = model.forward(data)
    assert output.shape == (2, 2)
    assert output.tolist() == [[0.0, 1000.0], [2000.0, 3000.0]]


def test_untrained_and_unbuilt_models_are_rejected():
    """Both preconditions raise before any layer runs."""
    unbuilt = build_model()
    unbuilt.rbm_layers = None
    with pytest.raises(ValueError, match="not built"):
        unbuilt.forward(np.zeros((1, 2), dtype=np.float32))

    untrained = build_model(trained=False)
    with pytest.raises(ValueError, match="not trained"):
        untrained.forward(np.zeros((1, 2), dtype=np.float32))
