"""Exercise the BM generation trainer with synchronous PyTorch loaders."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset


class FixedSampler:
    def solve(self, ising_matrix):
        spins = -np.ones((1, ising_matrix.shape[0]), dtype=int)
        spins[:, -1] = 1
        return spins


class RecordingPool:
    def __init__(self):
        self.events = []

    def map(self, worker, batches):
        return list(map(worker, batches))

    def close(self):
        self.events.append("close")

    def join(self):
        self.events.append("join")


class RecordingSaver:
    def __init__(self):
        self.saved_steps = []
        self.logged_steps = []

    def save_info(self, model, save_path, step, elapsed):
        self.saved_steps.append(step)

    def output_loss(self, step, kl_divergence, ncl, total):
        self.logged_steps.append(step)


class IntermittentDataset(IterableDataset):
    def __init__(self, nonempty_epochs):
        self.nonempty_epochs = nonempty_epochs
        self.epochs = 0

    def __len__(self):
        return 2

    def __iter__(self):
        self.epochs += 1
        if self.epochs > self.nonempty_epochs + 1:
            raise AssertionError("training retried an epoch that made no progress")
        yield torch.tensor([0.0, 1.0, 1.0])
        if self.epochs <= self.nonempty_epochs:
            yield torch.tensor([1.0, 0.0, 1.0])


class SingleBatchDataset(IterableDataset):
    def __iter__(self):
        yield torch.tensor([0.0, 1.0, 1.0])
        raise AssertionError("training fetched another batch after the last step")


class ZeroEstimateDataset(SingleBatchDataset):
    def __len__(self):
        return 0


@pytest.fixture
def training(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from example.bm_generation import trainer

    pool = RecordingPool()
    saver = RecordingSaver()
    monkeypatch.setattr(trainer.mp, "Pool", lambda processes: pool)
    monkeypatch.setattr(trainer.kw.common, "set_log_level", lambda level: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def make_trainer(data):
        return trainer.Trainer(
            data, saver, FixedSampler(), num_visible=3, num_hidden=2, num_output=1
        )

    return make_trainer, saver, pool


@pytest.mark.parametrize("nonempty_epochs", [0, 1])
def test_empty_epoch_uses_actual_batches(training, tmp_path, nonempty_epochs):
    make_trainer, saver, pool = training
    dataset = IntermittentDataset(nonempty_epochs)
    loader = DataLoader(dataset, batch_size=2, drop_last=True)
    assert len(loader) == 1

    with pytest.raises(ValueError, match="yielded no batches"):
        make_trainer(loader).train(max_steps=2, save_path=tmp_path)

    assert dataset.epochs == nonempty_epochs + 1
    assert saver.saved_steps == [0]
    assert saver.logged_steps == list(range(1, nonempty_epochs + 1))
    assert pool.events == ["close", "join"]


@pytest.mark.parametrize("dataset_type", [SingleBatchDataset, ZeroEstimateDataset])
def test_training_stops_fetching_at_step_limit(training, tmp_path, dataset_type):
    make_trainer, saver, pool = training
    loader = DataLoader(dataset_type(), batch_size=1)

    make_trainer(loader).train(max_steps=1, save_path=tmp_path)

    assert saver.logged_steps == [1]
    assert pool.events == ["close", "join"]


def test_reiterable_data_can_supply_more_than_one_epoch(training, tmp_path):
    make_trainer, saver, pool = training
    loader = DataLoader(torch.tensor([[0.0, 1.0, 1.0]]), batch_size=1)

    make_trainer(loader).train(max_steps=3, save_path=tmp_path)

    assert saver.logged_steps == [1, 2, 3]
    assert pool.events == ["close", "join"]


def test_zero_step_budget_does_not_iterate_data(training, tmp_path):
    make_trainer, saver, pool = training
    dataset = IntermittentDataset(nonempty_epochs=0)
    loader = DataLoader(dataset, batch_size=2, drop_last=True)

    make_trainer(loader).train(max_steps=0, save_path=tmp_path)

    assert dataset.epochs == 0
    assert saver.saved_steps == [0]
    assert saver.logged_steps == []
    assert pool.events == ["close", "join"]
