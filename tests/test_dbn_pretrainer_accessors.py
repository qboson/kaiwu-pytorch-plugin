"""Accessor delegation contract for the DBN pre-trainer."""

import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from dbn_trainer import DBNPretrainer  # noqa: E402


def build_pretrainer():
    """Builds only the wrapped-DBN reference the accessors read."""
    pretrainer = DBNPretrainer.__new__(DBNPretrainer)
    pretrainer._dbn = SimpleNamespace(
        num_layers=3,
        output_dim=64,
        device=torch.device("cpu"),
        get_rbm_layer=lambda index: ("layer", index),
    )
    return pretrainer


def test_accessors_delegate_to_the_wrapped_dbn():
    """Layer count, output width and device come from the wrapped model."""
    pretrainer = build_pretrainer()
    assert pretrainer._n_layers == 3
    assert pretrainer._output_dim == 64
    assert pretrainer.device == torch.device("cpu")
    assert pretrainer.get_rbm_layer(1) == ("layer", 1)
