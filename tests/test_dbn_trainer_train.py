"""Contract test for the ``DBNTrainer.train`` layer-wise loop."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from kaiwu.torch_plugin.dbn import UnsupervisedDBN

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "example" / "dbn_digits" / "dbn_trainer.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("dbn_digits_trainer_train_uut", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeLayer(SimpleNamespace):
    def __init__(self, visible, hidden):
        super().__init__(num_visible=visible, num_hidden=hidden)


class _FakeDBN(UnsupervisedDBN):
    def __init__(self):
        self.layers = []
        self.trained = False

    @property
    def num_layers(self):
        return len(self.layers)

    def create_rbm_layer(self, input_dim):
        self.layers = [_FakeLayer(input_dim, 3), _FakeLayer(3, 2)]

    def get_rbm_layer(self, index):
        return self.layers[index]

    def mark_as_trained(self):
        self.trained = True


def _trainer(module):
    return module.DBNTrainer(verbose=True, random_state=None)


def test_rejects_objects_that_are_not_a_dbn(monkeypatch):
    module = _load_trainer()
    monkeypatch.setattr(module, "UnsupervisedDBN", _FakeDBN)

    with pytest.raises(ValueError, match="dbn must be an instance"):
        _trainer(module).train(object(), np.zeros((4, 4)))


def test_pretrains_every_layer_in_sequence(monkeypatch, capsys):
    module = _load_trainer()
    monkeypatch.setattr(module, "UnsupervisedDBN", _FakeDBN)
    trainer = _trainer(module)
    dbn = _FakeDBN()
    calls = []

    def fake_layer_train(rbm, data_in, layer_idx):
        calls.append((rbm, data_in.copy(), layer_idx))
        return data_in[:, : rbm.num_hidden]

    monkeypatch.setattr(trainer, "_train_rbm_layer", fake_layer_train)
    data = np.arange(20, dtype=np.float64).reshape(5, 4)

    result = trainer.train(dbn, data)

    assert result is dbn
    assert trainer.dbn_ref is dbn
    assert [layer.num_visible for layer in dbn.layers] == [4, 3]
    assert len(calls) == 2
    first_rbm, first_data, first_index = calls[0]
    assert first_rbm is dbn.layers[0]
    assert first_index == 0
    assert first_data.dtype == np.float32
    assert first_data.shape == (5, 4)
    second_rbm, second_data, second_index = calls[1]
    assert second_rbm is dbn.layers[1]
    assert second_index == 1
    assert second_data.shape == (5, 3)
    assert dbn.trained is True
    output = capsys.readouterr().out
    assert "Pre-training RBM layer 1/2" in output
    assert "Pre-training RBM layer 2/2" in output


def test_reseeds_only_when_a_random_state_is_configured(monkeypatch):
    module = _load_trainer()
    monkeypatch.setattr(module, "UnsupervisedDBN", _FakeDBN)
    seeded = []
    trainer = _trainer(module)
    monkeypatch.setattr(trainer, "_train_rbm_layer", lambda rbm, data, index: data)
    monkeypatch.setattr(trainer, "_set_random_seed", lambda: seeded.append(True))

    trainer.train(_FakeDBN(), np.zeros((4, 4)))
    assert seeded == []

    seeded_trainer = module.DBNTrainer(verbose=False, random_state=7)
    monkeypatch.setattr(
        seeded_trainer, "_train_rbm_layer", lambda rbm, data, index: data
    )
    monkeypatch.setattr(
        seeded_trainer, "_set_random_seed", lambda: seeded.append(True)
    )
    seeded_trainer.train(_FakeDBN(), np.zeros((4, 4)))
    assert seeded == [True]
