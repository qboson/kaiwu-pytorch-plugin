"""Contract test for the ``DBNTrainer.__init__`` configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from kaiwu.classical import SimulatedAnnealingOptimizer

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "example" / "dbn_digits" / "dbn_trainer.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("dbn_digits_trainer_init_uut", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stores_the_configuration_and_builds_the_sa_sampler():
    module = _load_trainer()
    dbn_ref = object()

    trainer = module.DBNTrainer(
        learning_rate_rbm=0.05,
        n_epochs_rbm=3,
        batch_size=7,
        verbose=False,
        shuffle=False,
        drop_last=True,
        plot_img=True,
        random_state=11,
        dbn_ref=dbn_ref,
    )

    assert trainer.learning_rate_rbm == 0.05
    assert trainer.n_epochs_rbm == 3
    assert trainer.batch_size == 7
    assert trainer.verbose is False
    assert trainer.shuffle is False
    assert trainer.drop_last is True
    assert trainer.plot_img is True
    assert trainer.random_state == 11
    assert trainer.dbn_ref is dbn_ref
    assert isinstance(trainer.sampler, SimulatedAnnealingOptimizer)
    assert trainer.sampler._alpha == 0.999
    assert trainer.sampler._size_limit == 100


def test_documented_defaults_are_kept():
    module = _load_trainer()

    trainer = module.DBNTrainer()

    assert trainer.learning_rate_rbm == 0.1
    assert trainer.n_epochs_rbm == 10
    assert trainer.batch_size == 100
    assert trainer.verbose is True
    assert trainer.shuffle is True
    assert trainer.drop_last is False
    assert trainer.plot_img is False
    assert trainer.random_state is None
    assert trainer.dbn_ref is None
