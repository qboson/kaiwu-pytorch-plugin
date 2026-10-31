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


class _DatasetBoundaryReached(Exception):
    """Stop a valid CLI invocation before any data access or model work."""

    def __init__(self, options):
        super().__init__("synthetic dataset boundary reached")
        self.options = options


def _stop_before_dataset(**options):
    raise _DatasetBoundaryReached(options)


def _load_config():
    spec = importlib.util.spec_from_file_location("qvae_mnist_config_cell_uut", CONFIG)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_run_pipeline():
    names = ("model", "downstream", "utils", "utils.loadMNIST")
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in names}
    model_module = types.ModuleType("model")
    # Exercise the real constructor, rather than making an accepted argument
    # fail accidentally because ``object`` does not accept configuration kwargs.
    model_module.Config = _load_config().Config
    downstream_module = types.ModuleType("downstream")
    downstream_module.get_full_pipeline = lambda config: None
    utils_module = types.ModuleType("utils")
    utils_module.__path__ = []
    loadmnist_module = types.ModuleType("utils.loadMNIST")
    loadmnist_module.loadMNIST = _stop_before_dataset
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
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_config_rejects_the_unsupported_cell_mode():
    """Reject a configuration that the MNIST factory cannot construct."""
    config_module = _load_config()

    with pytest.raises(ValueError, match="Unsupported model type"):
        config_module.Config(model_type="CellQVAE")


def test_run_pipeline_rejects_the_unsupported_cell_mode(monkeypatch):
    """Reject the unsupported choice before configuration or dataset access."""
    module = _load_run_pipeline()
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--model-type", "CellQVAE"])

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert excinfo.value.code == 2


def test_supported_qvae_config_keeps_its_pipeline_options():
    """Keep the supported model and explicit pipeline options available."""
    config_module = _load_config()

    config = config_module.Config(model_type="QVAE", batch_size=7, name="mnist")

    assert config.type == "QVAE"
    assert config.batch_size == 7
    assert config.name == "mnist"


def test_run_pipeline_keeps_the_supported_qvae_mode(monkeypatch):
    """Accept the documented QVAE mode while stopping at the data boundary."""
    module = _load_run_pipeline()
    monkeypatch.setattr(
        sys, "argv", ["run_pipeline.py", "--model-type", "QVAE", "--batch-size", "7"]
    )

    with pytest.raises(_DatasetBoundaryReached) as excinfo:
        module.main()

    assert excinfo.value.options["name"] == "mnist"
    assert excinfo.value.options["batch_size"] == 7


def test_help_lists_only_the_supported_model_mode(monkeypatch, capsys):
    """Make CLI help describe the same supported model set as validation."""
    module = _load_run_pipeline()
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--model-type {QVAE}" in help_text
    assert "CellQVAE" not in help_text


def _ambient_modules(monkeypatch):
    expected = {
        "model": None,
        "downstream": types.ModuleType("ambient_downstream"),
        "utils": types.ModuleType("ambient_utils"),
        "utils.loadMNIST": None,
        "qvae_mnist_run_pipeline_cell_uut": types.ModuleType("ambient_same_spec"),
        "qvae_mnist_config_cell_uut": types.ModuleType("ambient_config_spec"),
    }
    for name, value in expected.items():
        monkeypatch.setitem(sys.modules, name, value)
    return expected


def test_loader_restores_existing_modules_and_path(monkeypatch):
    """Preserve module identities, blocked None entries, and the import path."""
    expected = _ambient_modules(monkeypatch)
    path_object = sys.path
    path_entries = list(sys.path)

    loaded = _load_run_pipeline()

    assert all(sys.modules[name] is value for name, value in expected.items())
    assert loaded is not expected["qvae_mnist_run_pipeline_cell_uut"]
    assert sys.path is path_object
    assert sys.path == path_entries


def test_loader_restores_modules_when_source_import_fails(monkeypatch, tmp_path):
    """Restore the same import state when the controlled source raises."""
    expected = _ambient_modules(monkeypatch)
    path_object = sys.path
    path_entries = list(sys.path)
    failing_source = tmp_path / "controlled_import_failure.py"
    failing_source.write_text(
        'raise RuntimeError("synthetic import failure")\n', encoding="utf-8"
    )
    monkeypatch.setitem(globals(), "RUN_PIPELINE", failing_source)

    with pytest.raises(RuntimeError, match="synthetic import failure"):
        _load_run_pipeline()

    assert all(sys.modules[name] is value for name, value in expected.items())
    assert sys.path is path_object
    assert sys.path == path_entries
