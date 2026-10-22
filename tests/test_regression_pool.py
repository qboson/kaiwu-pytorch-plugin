"""Real sampling worker processes must be reaped when training fails."""
import importlib
from pathlib import Path
import sys
import numpy as np
import pytest
import torch


@pytest.fixture
def trainer_module(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "example/bm_generation/trainer.py"
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.delitem(sys.modules, "trainer", raising=False)
    module = importlib.import_module("trainer")
    monkeypatch.setitem(sys.modules, "trainer", module)
    return module


class Saver:
    def save_info(self, *args):
        pass

    def output_loss(self, *args):
        pass


class LocalSampler:
    def solve(self, matrix):
        return np.ones((1, matrix.shape[0]))


def test_successful_training_uses_and_reaps_workers(trainer_module, monkeypatch, tmp_path):
    pools = []
    # Spawn is safe after other tests have initialized PyTorch's thread pools.
    original_pool = trainer_module.mp.get_context("spawn").Pool

    def tracked_pool(*args, **kwargs):
        pool = original_pool(*args, **kwargs)
        pools.append(pool)
        return pool

    losses = []
    saver = Saver()
    monkeypatch.setattr(saver, "output_loss", lambda *args: losses.append(args))
    monkeypatch.setattr(trainer_module.mp, "Pool", tracked_pool)
    trainer = trainer_module.Trainer([torch.zeros(4, 2)], saver, LocalSampler(),
                                     num_visible=2, num_hidden=1, num_output=1)
    try:
        trainer.train(2, str(tmp_path), num_processes=2)
        assert [row[0] for row in losses] == [1, 2]
        assert np.isfinite(np.asarray(losses)).all()
        assert all(worker.exitcode is not None for worker in pools[0]._pool)
        assert all(not worker.is_alive() for worker in pools[0]._pool)
    finally:
        for pool in pools:
            pool.terminate()
            pool.join()


@pytest.mark.parametrize("failure", ["sampling", "data"])
def test_training_exception_reaps_actual_pool(trainer_module, monkeypatch, tmp_path, failure):
    module = trainer_module
    pools = []
    original_pool = module.mp.get_context("spawn").Pool

    def tracked_pool(*args, **kwargs):
        pool = original_pool(*args, **kwargs)
        pools.append(pool)
        return pool

    class BrokenData:
        def __iter__(self):
            raise RuntimeError("data failure")

    def broken_sample(*args, **kwargs):
        raise RuntimeError("sampling failure")

    monkeypatch.setattr(module.mp, "Pool", tracked_pool)
    trainer = module.Trainer(BrokenData() if failure == "data" else [torch.zeros(2, 2)],
                             Saver(), object(), num_visible=2, num_hidden=1, num_output=1)
    monkeypatch.setattr(trainer.bm_net, "sample", broken_sample)
    try:
        with pytest.raises(RuntimeError, match=f"{failure} failure"):
            trainer.train(1, str(tmp_path), num_processes=2)
        assert len(pools) == 1
        assert all(worker.exitcode is not None for worker in pools[0]._pool)
        assert all(not worker.is_alive() for worker in pools[0]._pool)
    finally:
        for pool in pools:
            pool.terminate()
            pool.join()
