"""Training-configuration contract for the DBN trainer."""

import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from dbn_trainer import DBNTrainer  # noqa: E402


def build_trainer():
    """Builds a trainer with explicit settings and a device reference."""
    return DBNTrainer(
        learning_rate_rbm=0.05,
        n_epochs_rbm=3,
        batch_size=16,
        verbose=False,
        shuffle=False,
        drop_last=True,
        plot_img=True,
        random_state=7,
        dbn_ref=SimpleNamespace(device="cpu"),
    )


def test_configuration_reports_every_training_knob():
    """The dict mirrors the constructor arguments."""
    config = build_trainer().get_training_config()
    assert config == {
        "learning_rate_rbm": 0.05,
        "n_epochs_rbm": 3,
        "batch_size": 16,
        "verbose": False,
        "shuffle": False,
        "drop_last": True,
        "plot_img": True,
        "random_state": 7,
        "device": "cpu",
    }


def test_missing_device_reference_is_reported_as_unknown():
    """Without a dbn reference the device entry falls back to ``unknown``."""
    trainer = DBNTrainer(dbn_ref=None)
    assert trainer.get_training_config()["device"] == "unknown"


def test_defaults_are_the_documented_ones():
    """The default trainer settings stay at the documented values."""
    config = DBNTrainer(dbn_ref=SimpleNamespace(device="cuda:0")).get_training_config()
    assert config["learning_rate_rbm"] == 0.1
    assert config["n_epochs_rbm"] == 10
    assert config["batch_size"] == 100
    assert config["verbose"] is True
    assert config["shuffle"] is True
    assert config["drop_last"] is False
    assert config["plot_img"] is False
    assert config["random_state"] is None
    assert config["device"] == "cuda:0"
