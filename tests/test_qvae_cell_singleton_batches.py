"""Exercise the single-cell example's actual loaders and BatchNorm networks."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


EXAMPLE = Path(__file__).resolve().parents[1] / "example" / "qvae_cell"


@contextmanager
def isolated_cell_modules(monkeypatch):
    """Load the example without colliding with another example's models module."""
    monkeypatch.syspath_prepend(str(EXAMPLE))
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in ("models", "batching")}
    try:
        for name in saved:
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location("qvae_cell_test_trainer", EXAMPLE / "trainer.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        assert Path(trainer.__file__).resolve() == EXAMPLE / "trainer.py"
        for name in saved:
            assert Path(sys.modules[name].__file__).resolve() == EXAMPLE / (name + ".py")
        yield trainer, sys.modules["models"]
    finally:
        for name, original in saved.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture
def cell_modules(monkeypatch):
    with isolated_cell_modules(monkeypatch) as modules:
        yield modules


@pytest.mark.parametrize("cached", [False, True])
def test_example_import_restores_module_cache(monkeypatch, cached):
    """Both absent and unrelated cached modules survive the fixture unchanged."""
    for name in ("models", "batching"):
        if cached:
            monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__="unrelated.py"))
        else:
            monkeypatch.delitem(sys.modules, name, raising=False)
    before = {name: sys.modules.get(name) for name in ("models", "batching")}
    with isolated_cell_modules(monkeypatch):
        assert all(sys.modules[name] is not before[name] for name in before)
    for name, original in before.items():
        if cached:
            assert sys.modules[name] is original
        else:
            assert name not in sys.modules


def make_loaders(trainer_module, count, batch_size, normalization="batch", load_weights=False):
    """Reserve one validation observation and expose row identities as labels."""
    args = SimpleNamespace(
        seed=42, val_percentage=0.1, batch_size=batch_size,
        normalization_method=normalization,
        load_weights=load_weights,
    )
    rows = np.arange((count + 1) * 4, dtype=np.float32).reshape(count + 1, 4)
    trainer = trainer_module.Trainer(args, torch.device("cpu"))
    return trainer.make_loaders(rows, np.arange(count + 1))


@pytest.mark.parametrize("count,batch_size", [(3, 2), (5, 2), (9, 4), (6, 4), (8, 4)])
def test_all_training_rows_reach_real_batchnorm_networks(cell_modules, count, batch_size):
    """Every retained observation participates once in a finite training step."""
    trainer_module, models = cell_modules
    train, validation, evaluation = make_loaders(trainer_module, count, batch_size)
    network = nn.Sequential(
        models.QVAEEncoder(4, 6, 2, "batch"),
        models.QVAEDecoder(2, 6, 4, "batch"),
    ).train()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001)
    seen = []
    batches = list(train)
    assert len(train) == len(batches)
    for values, identities in batches:
        seen.extend(identities.tolist())
        optimizer.zero_grad()
        loss = network(values).square().mean()
        assert 2 <= len(values) <= batch_size + 1
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in network.parameters())
        optimizer.step()
    expected = train.dataset.tensors[1].tolist()
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == count
    assert len(validation.dataset) == 1
    assert len(evaluation.dataset) == count + 1
    assert [len(values) for values, _ in validation] == [1]
    assert all(module.training for module in network.modules())
    assert all(module.num_batches_tracked == len(batches) for module in network.modules()
               if isinstance(module, nn.BatchNorm1d))


def test_layer_normalization_keeps_singleton_tail(cell_modules):
    """LayerNorm can train singleton batches and keeps the existing grouping."""
    trainer_module, models = cell_modules
    train, _, _ = make_loaders(trainer_module, 5, 2, "layer")
    assert [len(values) for values, _ in train] == [2, 2, 1]
    network = models.QVAEEncoder(4, 6, 2, "layer").train()
    assert torch.isfinite(network(next(reversed(list(train)))[0])).all()


def test_training_shuffle_repeats_with_the_same_seed(cell_modules):
    """The repair preserves PyTorch's caller-controlled random stream."""
    trainer_module, _ = cell_modules
    train, _, _ = make_loaders(trainer_module, 5, 2)
    torch.manual_seed(123)
    first = [ids.tolist() for _, ids in train]
    torch.manual_seed(123)
    assert [ids.tolist() for _, ids in train] == first


def test_sampler_preserves_the_original_shuffled_index_order(cell_modules):
    trainer_module, _ = cell_modules
    repaired, _, _ = make_loaders(trainer_module, 9, 4)
    original, _, _ = make_loaders(trainer_module, 9, 4, "layer")
    torch.manual_seed(321)
    original_ids = [index for _, ids in original for index in ids.tolist()]
    torch.manual_seed(321)
    assert [index for _, ids in repaired for index in ids.tolist()] == original_ids


def test_sampler_boundary_grid(cell_modules):
    """Check all indices, lengths, and batch bounds across many remainders."""
    trainer_module, _ = cell_modules
    sampler_class = trainer_module.SingletonSafeBatchSampler
    for batch_size in range(2, 18):
        for count in range(2, 68):
            sampler = sampler_class(torch.utils.data.SequentialSampler(range(count)), batch_size)
            for _ in range(2):
                batches = list(sampler)
                assert len(batches) == len(sampler)
                assert [index for batch in batches for index in batch] == list(range(count))
                assert all(2 <= len(batch) <= batch_size + 1 for batch in batches)


def test_batch_normalization_rejects_batch_size_one(cell_modules):
    trainer_module, _ = cell_modules
    with pytest.raises(ValueError, match="batch_size.*at least 2"):
        make_loaders(trainer_module, 3, 1)


def test_batch_normalization_rejects_a_one_observation_training_split(cell_modules):
    trainer_module, _ = cell_modules
    with pytest.raises(ValueError, match="at least two training observations"):
        make_loaders(trainer_module, 1, 2)


@pytest.mark.parametrize("count,batch_size", [(1, 1), (3, 2)])
def test_load_weights_keeps_inference_batching(cell_modules, count, batch_size):
    """The CLI's load-only path uses BatchNorm running statistics."""
    trainer_module, models = cell_modules
    train, _, evaluation = make_loaders(trainer_module, count, batch_size, load_weights=True)
    assert max(len(values) for values, _ in train) <= batch_size
    network = models.QVAEEncoder(4, 6, 2, "batch").eval()
    for values, _ in evaluation:
        assert torch.isfinite(network(values)).all()
