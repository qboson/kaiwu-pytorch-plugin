"""Contract test for rejecting the unsupported CellQVAE mode in qvae_mnist."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example" / "qvae_mnist"
CONFIG = EXAMPLE / "model" / "config.py"
RUN_PIPELINE = EXAMPLE / "run_pipeline.py"


def _load_config():
    spec = importlib.util.spec_from_file_location("qvae_mnist_config_cell_uut", CONFIG)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_run_pipeline():
    names = ("model", "downstream", "utils", "utils.loadMNIST")
    saved = {name: sys.modules.get(name) for name in names}
    model_module = types.ModuleType("model")
    model_module.Config = object
    downstream_module = types.ModuleType("downstream")
    downstream_module.get_full_pipeline = lambda config: None
    utils_module = types.ModuleType("utils")
    utils_module.__path__ = []
    loadmnist_module = types.ModuleType("utils.loadMNIST")
    loadmnist_module.loadMNIST = lambda **kwargs: ([], [])
    sys.modules.update(
        {
            "model": model_module,
            "downstream": downstream_module,
            "utils": utils_module,
            "utils.loadMNIST": loadmnist_module,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "qvae_mnist_run_pipeline_cell_uut", RUN_PIPELINE
        )
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


def test_config_rejects_the_unsupported_cell_mode():
    config_module = _load_config()

    with pytest.raises(ValueError, match="Unsupported model type"):
        config_module.Config(model_type="CellQVAE")


def test_run_pipeline_rejects_the_unsupported_cell_mode(monkeypatch):
    module = _load_run_pipeline()
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--model-type", "CellQVAE"])

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert excinfo.value.code == 2
