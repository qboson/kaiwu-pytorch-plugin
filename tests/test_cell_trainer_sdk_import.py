"""The Cell trainer must import against the project's pinned Kaiwu SDK."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.cim import CIMOptimizer, PrecisionReducer


CELL_DIRECTORY = Path(__file__).resolve().parents[1] / "example" / "qvae_cell"


def load_example_module(name, filename):
    """Load the real example source without relying on the current directory."""
    spec = importlib.util.spec_from_file_location(name, CELL_DIRECTORY / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def offline_sampler():
    """Construct the official local optimizer without solving or authenticating."""
    return SimulatedAnnealingOptimizer(
        initial_temperature=10,
        alpha=0.5,
        cutoff_temperature=1,
        iterations_per_t=1,
        size_limit=2,
        rand_seed=0,
    )


def test_cell_trainer_import_and_local_sampler_with_real_sdk(monkeypatch):
    """CLI and notebook Trainer imports also work when selecting the local SA path."""
    pytest.importorskip("tqdm", reason="Cell example's optional training dependency")
    pytest.importorskip("sklearn.model_selection", reason="Cell example's optional data dependency")
    models = load_example_module("_cell_sdk_models", "models.py")
    monkeypatch.setitem(sys.modules, "models", models)

    trainer_module = load_example_module("_cell_sdk_trainer", "trainer.py")

    assert Path(trainer_module.__file__).resolve() == CELL_DIRECTORY / "trainer.py"
    assert trainer_module.PrecisionReducer is PrecisionReducer
    assert trainer_module.CIMOptimizer is CIMOptimizer
    args = SimpleNamespace(
        sampler_type="sa",
        sa_initial_temperature=10,
        sa_alpha=0.5,
        sa_cutoff_temperature=1,
        sa_iterations_per_t=1,
        sa_size_limit=2,
        sa_rand_seed=0,
    )
    trainer = trainer_module.Trainer(args, torch.device("cpu"))
    assert isinstance(trainer.create_sampler(), SimulatedAnnealingOptimizer)


def test_real_precision_reducer_accepts_the_cells_constructor_options():
    """The pinned SDK's actual CIM wrapper can be constructed without remote work."""
    reducer = PrecisionReducer(
        offline_sampler(),
        precision=8,
        truncated_precision=10,
        target_bits=32,
        only_feasible_solution=False,
    )

    assert isinstance(reducer, PrecisionReducer)
