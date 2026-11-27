# -*- coding: utf-8 -*-
"""DBN layer activation inspection accepts torch tensors as well as arrays."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "example" / "dbn_digits"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import supervised_dbn_digits as sdd  # noqa: E402  pylint: disable=wrong-import-position


class _FakeRBM:
    num_visible = 3

    def get_hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([inputs, torch.zeros(inputs.size(0), 1)], dim=1)


def _receiver():
    rbm = _FakeRBM()
    dbn = SimpleNamespace(
        _n_layers=1,
        device=torch.device("cpu"),
        get_rbm_layer=lambda index: rbm if index == 0 else None,
    )
    return SimpleNamespace(unsupervised_dbn=dbn)


def test_torch_input_is_supported() -> None:
    inputs = torch.randn(2, 3)

    activations = sdd.AbstractSupervisedDBNClassifier.get_layer_activations(
        _receiver(), inputs
    )

    assert activations.shape == (2, 1)
    assert np.isfinite(activations).all()


def test_numpy_input_is_unchanged() -> None:
    inputs = np.ones((2, 3), dtype=np.float32)

    activations = sdd.AbstractSupervisedDBNClassifier.get_layer_activations(
        _receiver(), inputs
    )

    assert activations.shape == (2, 1)
    assert np.isfinite(activations).all()


def test_all_layers_returns_a_mapping() -> None:
    activations = sdd.AbstractSupervisedDBNClassifier.get_layer_activations(
        _receiver(), torch.zeros(2, 3), return_all_layers=True
    )

    assert set(activations) == {0}
    assert activations[0].shape == (2, 1)
