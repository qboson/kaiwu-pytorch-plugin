"""The legacy momentum setting must not silently promise an Adam control."""
import importlib.util
from pathlib import Path
import warnings
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    'bm_example', Path(__file__).resolve().parents[1] / 'example/bm_generation/trainer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_nonzero_legacy_momentum_warns_at_call_site():
    trainer = module.Trainer([], Mock(), Mock(), num_visible=2, num_hidden=1)
    with pytest.warns(FutureWarning, match='Adam') as records:
        trainer.set_learning_parameters(.01, .02, .9)
    assert records[0].filename == __file__
    assert trainer.learning_parameters == {
        'learning_rate': .01, 'weight_decay_rate': .02, 'momentum_rate': .9}


def test_zero_legacy_momentum_is_compatible():
    trainer = module.Trainer([], Mock(), Mock(), num_visible=2, num_hidden=1)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        trainer.set_learning_parameters(.01, .02, 0.)
    assert not records
