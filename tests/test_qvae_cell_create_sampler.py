"""Sampler-selection contract for the qvae_cell trainer."""

import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_cell"))
pre = importlib.import_module("kaiwu.preprocess")
if not hasattr(pre, "PrecisionReducer"):
    pre.PrecisionReducer = type(
        "PrecisionReducer", (), {"__init__": lambda self, *args, **kwargs: None}
    )

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_trainer_sampler", os.path.join(REPO_ROOT, "example", "qvae_cell", "trainer.py")
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer


class RecordingSampler:
    """Records the SA constructor kwargs."""

    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingSampler.calls.append(kwargs)


def build_trainer(sampler_type):
    """Builds only the attributes ``create_sampler`` reads."""
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        sampler_type=sampler_type,
        sa_initial_temperature=2.0,
        sa_alpha=0.9,
        sa_cutoff_temperature=0.01,
        sa_iterations_per_t=3,
        sa_size_limit=7,
        sa_rand_seed=11,
    )
    return trainer


def test_sa_sampler_receives_every_configured_knob(monkeypatch):
    """The SA branch forwards temperature, cooling, iterations and seed."""
    RecordingSampler.calls = []
    monkeypatch.setattr(
        trainer_module, "SimulatedAnnealingOptimizer", RecordingSampler
    )
    sampler = build_trainer("sa").create_sampler()
    assert isinstance(sampler, RecordingSampler)
    assert RecordingSampler.calls == [
        {
            "initial_temperature": 2.0,
            "alpha": 0.9,
            "cutoff_temperature": 0.01,
            "iterations_per_t": 3,
            "size_limit": 7,
            "rand_seed": 11,
        }
    ]


def test_unknown_sampler_type_is_rejected(monkeypatch):
    """Anything outside sa/cim fails with the offending value."""
    monkeypatch.setattr(
        trainer_module, "SimulatedAnnealingOptimizer", RecordingSampler
    )
    with pytest.raises(ValueError, match="Unsupported sampler_type: bogus"):
        build_trainer("bogus").create_sampler()
