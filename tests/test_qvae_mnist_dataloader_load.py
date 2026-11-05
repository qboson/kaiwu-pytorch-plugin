"""Contract test for ``MNISTDataLoader.load`` argument forwarding."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOADER = REPO / "example" / "qvae_mnist" / "utils" / "loadMNIST.py"


def _load_loader():
    spec = importlib.util.spec_from_file_location("qvae_mnist_loadmnist_load_uut", LOADER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_forwards_the_configured_arguments_and_stores_the_loaders(monkeypatch):
    loader_module = _load_loader()
    calls = []
    train_loader = object()
    test_loader = object()

    def fake_load_mnist(**kwargs):
        calls.append(kwargs)
        return train_loader, test_loader

    monkeypatch.setattr(loader_module, "loadMNIST", fake_load_mnist)
    loader = loader_module.MNISTDataLoader(
        batch_size=7,
        num_evts_train=11,
        num_evts_test=13,
        data_path="./custom",
    )

    result = loader.load()

    assert calls == [
        {
            "batch_size": 7,
            "num_evts_train": 11,
            "num_evts_test": 13,
            "data_path": "./custom",
        }
    ]
    assert result == (train_loader, test_loader)
    assert loader.train_loader is train_loader
    assert loader.test_loader is test_loader
