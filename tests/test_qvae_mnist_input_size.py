"""Contract test for ``MNISTDataLoader.get_input_size``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
LOADER = REPO / "example" / "qvae_mnist" / "utils" / "loadMNIST.py"


def _load_loader():
    spec = importlib.util.spec_from_file_location("qvae_mnist_loadmnist_uut", LOADER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def test_rejects_an_unloaded_loader():
    loader_module = _load_loader()
    loader = loader_module.MNISTDataLoader()

    with pytest.raises(ValueError, match="Call load"):
        loader.get_input_size()


def test_reads_the_feature_dimension_from_the_first_batch():
    loader_module = _load_loader()
    loader = loader_module.MNISTDataLoader()
    loader.train_loader = _FakeLoader(
        [(torch.zeros(5, 784), torch.zeros(5, dtype=torch.long))]
    )

    assert loader.get_input_size() == 784


def test_reports_one_dimension_per_multi_input_tensor():
    loader_module = _load_loader()
    loader = loader_module.MNISTDataLoader()
    loader.train_loader = _FakeLoader(
        [
            (
                [torch.zeros(5, 10), torch.zeros(5, 20)],
                torch.zeros(5, dtype=torch.long),
            )
        ]
    )

    assert loader.get_input_size() == [10, 20]
