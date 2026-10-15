"""BatchNorm training must retain every observation without singleton batches."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def trainer_module(monkeypatch):
    import kaiwu
    from example.qvae_cell import models
    # main has an unrelated SDK import issue tracked by #202. Use the real
    # SDK class only to import the trainer; these tests never create a sampler.
    monkeypatch.setattr(kaiwu.preprocess, "PrecisionReducer",
                        kaiwu.cim.PrecisionReducer, raising=False)
    monkeypatch.setitem(sys.modules, "models", models)
    path = Path(__file__).resolve().parents[1] / "example/qvae_cell/trainer.py"
    spec = importlib.util.spec_from_file_location("cell_tail_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n_cells,batch_size", [(12, 4), (10, 3), (6, 4), (16, 4)])
def test_batchnorm_training_retains_tail_cells(trainer_module, n_cells, batch_size):
    from example.qvae_cell.models import QVAEEncoder, QVAEDecoder
    args = SimpleNamespace(val_percentage=.25, seed=42, batch_size=batch_size,
                           normalization_method="batch")
    trainer = trainer_module.Trainer(args, torch.device("cpu"))
    x = np.arange(n_cells * 3, dtype=np.float32).reshape(n_cells, 3)
    train, val, all_cells = trainer.make_loaders(x, np.zeros(n_cells, dtype=int))
    encoder = QVAEEncoder(3, 4, 2, "batch").train()
    decoder = QVAEDecoder(2, 4, 3, "batch").train()
    seen = []
    sizes = []
    for batch, _ in train:
        sizes.append(len(batch))
        decoder(encoder(batch)).sum().backward()
        seen.extend(batch[:, 0].tolist())
    assert min(sizes) >= 2
    assert max(sizes) <= batch_size + 1
    assert len(sizes) == len(train)
    assert sorted(seen) == sorted(train.dataset.tensors[0][:, 0].tolist())
    assert len(seen) + len(val.dataset) == len(all_cells.dataset) == n_cells


def test_layernorm_keeps_existing_singleton_batches(trainer_module):
    args = SimpleNamespace(val_percentage=.25, seed=42, batch_size=4,
                           normalization_method="layer")
    train, _, _ = trainer_module.Trainer(args, "cpu").make_loaders(
        np.zeros((12, 3)), np.zeros(12, dtype=int))
    assert [len(x) for x, _ in train] == [4, 4, 1]


def test_batchnorm_rejects_unsatisfiable_batch_size(trainer_module):
    args = SimpleNamespace(val_percentage=.25, seed=42, batch_size=1,
                           normalization_method="batch")
    with pytest.raises(ValueError, match="BatchNorm.*at least 2"):
        trainer_module.Trainer(args, "cpu").make_loaders(
            np.zeros((12, 3)), np.zeros(12, dtype=int))
