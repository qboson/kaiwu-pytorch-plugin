"""Contract test for the ``ModelTuner`` input-size and dataset-mean getters."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
TUNER = REPO / "example" / "qvae_mnist" / "trainer" / "model_tuner.py"


def _load_tuner():
    names = ("utils", "utils.logging", "utils.exception")
    saved = {name: sys.modules.get(name) for name in names}
    utils_module = types.ModuleType("utils")
    utils_module.__path__ = []
    logging_module = types.ModuleType("utils.logging")
    logging_module.get_logger = logging.getLogger
    exception_module = types.ModuleType("utils.exception")
    exception_module.ValueError = ValueError
    exception_module.TypeError = TypeError
    sys.modules.update(
        {
            "utils": utils_module,
            "utils.logging": logging_module,
            "utils.exception": exception_module,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location("qvae_mnist_tuner_uut", TUNER)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _Dataset:
    def __iter__(self):
        return iter(
            [
                ([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])], 0),
                ([torch.tensor([3.0, 2.0, 1.0]), torch.tensor([6.0, 5.0])], 1),
            ]
        )


class _Loader:
    def __init__(self, input_size):
        self.dataset = _Dataset()
        self._input_size = input_size

    def get_input_size(self):
        return self._input_size


def test_get_input_dimension_delegates_to_the_loader():
    tuner_module = _load_tuner()
    tuner = tuner_module.ModelTuner()
    tuner.train_loader = _Loader([3, 2])

    assert tuner.get_input_dimension() == [3, 2]


def test_get_input_dimension_rejects_an_empty_train_loader():
    tuner_module = _load_tuner()
    tuner = tuner_module.ModelTuner()

    with pytest.raises(AssertionError, match="empty train loader"):
        tuner.get_input_dimension()


def test_get_train_dataset_mean_averages_every_input_layer():
    tuner_module = _load_tuner()
    tuner = tuner_module.ModelTuner()
    tuner.train_loader = _Loader([3, 2])

    means = tuner.get_train_dataset_mean()

    assert len(means) == 2
    assert means[0].shape == (1, 3)
    assert means[1].shape == (1, 2)
    assert torch.allclose(means[0], torch.tensor([[2.0, 2.0, 2.0]]))
    assert torch.allclose(means[1], torch.tensor([[5.0, 5.0]]))


def test_get_train_dataset_mean_rejects_an_empty_train_loader():
    tuner_module = _load_tuner()
    tuner = tuner_module.ModelTuner()

    with pytest.raises(AssertionError, match="empty train loader"):
        tuner.get_train_dataset_mean()
