"""Constructor contract for the BM generation trainer."""

import importlib.util
import os

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "bm_generation_trainer_init",
    os.path.join(REPO_ROOT, "example", "bm_generation", "trainer.py"),
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer


def build_trainer():
    """Builds a trainer with explicit layer sizes."""
    return Trainer(
        data=[],
        saver=object(),
        worker=object(),
        num_visible=7,
        num_hidden=3,
        num_output=2,
    )


def test_layer_sizes_and_references_are_stored():
    """Visible/hidden/output sizes are kept and the BM has one node per unit."""
    trainer = build_trainer()
    assert (trainer.num_visible, trainer.num_hidden, trainer.num_output) == (7, 3, 2)
    assert trainer.bm_net.num_nodes == 10
    assert trainer.bm_net.device.type == "cpu"
    assert trainer.bm_net_cpu is trainer.bm_net


def test_default_hyperparameters_are_documented_values():
    """The learning and cost dictionaries start at the documented defaults."""
    trainer = build_trainer()
    assert trainer.learning_parameters == {
        "learning_rate": 0.001,
        "weight_decay_rate": 0.0,
        "momentum_rate": 0.0,
    }
    assert trainer.cost_param == {"alpha": 0.5, "beta": 0.5}
