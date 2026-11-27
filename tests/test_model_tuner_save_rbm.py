"""Regression test for ``ModelTuner.save_rbm`` attribute naming.

``save_rbm`` printed and serialized ``self._model.prior``, but the QVAE model
exposes its Boltzmann machine as ``self.bm`` (no ``prior`` attribute exists
anywhere on ``kaiwu.torch_plugin.QVAE`` or the MNIST example subclass), so the
method crashed with ``AttributeError`` before writing anything.
"""

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
QVAE_MNIST_DIR = REPO_ROOT / "example" / "qvae_mnist"


def _load_model_tuner(monkeypatch):
    """Loads trainer/model_tuner.py with its example-local utils stubbed."""
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(QVAE_MNIST_DIR / "utils")]
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)

    logging_stub = types.ModuleType("utils.logging")
    logging_stub.get_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "utils.logging", logging_stub)

    exception_stub = types.ModuleType("utils.exception")
    exception_stub.ValueError = ValueError
    exception_stub.TypeError = TypeError
    monkeypatch.setitem(sys.modules, "utils.exception", exception_stub)

    spec = importlib.util.spec_from_file_location(
        "model_tuner_under_test", QVAE_MNIST_DIR / "trainer" / "model_tuner.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _QVAEWithBM(torch.nn.Module):
    """Model exposing exactly the public surface QVAE provides."""

    def __init__(self):
        super().__init__()
        self.bm = torch.nn.Linear(3, 3)


def test_save_rbm_persists_the_boltzmann_machine(monkeypatch, tmp_path):
    module = _load_model_tuner(monkeypatch)
    tuner = module.ModelTuner()
    tuner.outpath = str(tmp_path)
    model = _QVAEWithBM()
    tuner.register_model(model)

    tuner.save_rbm(config_string="unit")

    saved = tmp_path / "rbm_unit.pt"
    assert saved.is_file()
    restored = torch.load(saved, weights_only=False)
    assert isinstance(restored, torch.nn.Linear)
    assert torch.equal(
        restored.weight.data,
        model.bm.weight.data,
    )


def test_save_model_still_saves_the_full_state_dict(monkeypatch, tmp_path):
    module = _load_model_tuner(monkeypatch)
    tuner = module.ModelTuner()
    tuner.outpath = str(tmp_path)
    tuner.register_model(_QVAEWithBM())

    tuner.save_model(config_string="unit")

    saved = tmp_path / "model_unit.pt"
    assert saved.is_file()
    state_dict = torch.load(saved, weights_only=False)
    assert "bm.weight" in state_dict


def test_qvae_exposes_bm_not_prior():
    """The real plugin class must keep offering the attribute we save."""
    from kaiwu.torch_plugin import QVAE

    assert hasattr(QVAE, "_create_bm")
    assert not hasattr(QVAE, "prior")
