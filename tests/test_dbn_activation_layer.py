"""Activation-selection contract for the supervised DBN example."""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


def build_model(activation):
    """Builds only the attribute ``_get_activation_layer`` reads."""
    model = SupervisedDBNClassification.__new__(SupervisedDBNClassification)
    model.activation_function = activation
    return model


def test_supported_activations_map_to_modules():
    """The two documented names map to the matching torch modules."""
    assert isinstance(build_model("sigmoid")._get_activation_layer(), torch.nn.Sigmoid)
    assert isinstance(build_model("relu")._get_activation_layer(), torch.nn.ReLU)


def test_unknown_activation_is_rejected_with_the_name():
    """Anything outside the map fails with the offending name."""
    with pytest.raises(ValueError, match="Unsupported activation: tanh"):
        build_model("tanh")._get_activation_layer()
