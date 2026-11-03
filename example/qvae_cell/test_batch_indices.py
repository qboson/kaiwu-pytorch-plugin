"""Regression checks for batch labels in the single-cell example."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


def _trainer_class(monkeypatch):
    kaiwu = ModuleType("kaiwu")
    kaiwu.classical = ModuleType("kaiwu.classical")
    kaiwu.cim = ModuleType("kaiwu.cim")
    kaiwu.preprocess = ModuleType("kaiwu.preprocess")
    kaiwu.torch_plugin = ModuleType("kaiwu.torch_plugin")
    models = ModuleType("models")
    for module, names in (
        (kaiwu.classical, ("SimulatedAnnealingOptimizer",)),
        (kaiwu.cim, ("CIMOptimizer",)),
        (kaiwu.preprocess, ("PrecisionReducer",)),
        (kaiwu.torch_plugin, ("BoltzmannMachine",)),
        (models, ("CellQVAE", "QVAEDecoder", "QVAEEncoder")),
    ):
        for name in names:
            setattr(module, name, object)
    modules = {
        "kaiwu": kaiwu,
        "kaiwu.classical": kaiwu.classical,
        "kaiwu.cim": kaiwu.cim,
        "kaiwu.preprocess": kaiwu.preprocess,
        "kaiwu.torch_plugin": kaiwu.torch_plugin,
        "models": models,
    }
    spec = importlib.util.spec_from_file_location("qvae_cell_trainer", Path(__file__).with_name("trainer.py"))
    module = importlib.util.module_from_spec(spec)
    for name, stub in modules.items():
        monkeypatch.setitem(sys.modules, name, stub)
    spec.loader.exec_module(module)
    return module.Trainer


def test_missing_batch_label_is_rejected_before_one_hot(monkeypatch):
    trainer = _trainer_class(monkeypatch)(SimpleNamespace(), "cpu")
    adata = SimpleNamespace(obs=pd.DataFrame({"batch": ["first", None, "second"]}))

    with pytest.raises(ValueError, match="missing batch labels"):
        trainer.batch_indices(adata, "batch")


def test_complete_batch_labels_still_have_nonnegative_codes(monkeypatch):
    trainer = _trainer_class(monkeypatch)(SimpleNamespace(), "cpu")
    adata = SimpleNamespace(obs=pd.DataFrame({"batch": ["first", "second", "first"]}))

    assert trainer.batch_indices(adata, "batch").tolist() == [0, 1, 0]
    assert trainer.n_batches == 2
