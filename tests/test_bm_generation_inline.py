"""Exercise the default single-process BM training route without a license."""

import numpy as np
import pytest
import torch

pytest.importorskip("matplotlib")

from example.bm_generation import trainer


def test_single_process_uses_the_caller_sampler_without_spawning(monkeypatch):
    class LocalSampler:
        def __init__(self):
            self.widths = []

        def solve(self, matrix):
            self.widths.append(matrix.shape[0])
            return np.ones((1, matrix.shape[0]), dtype=np.int8)

    class RecordingSaver:
        losses = []

        def save_info(self, *args):
            pass

        def output_loss(self, step, kl_div, ncl, objective):
            self.losses.append((step, float(objective)))

    def unexpected(*args, **kwargs):
        pytest.fail("Inline sampling must not start a pool or change the SDK log level")

    monkeypatch.setattr(trainer.mp, "Pool", unexpected)
    monkeypatch.setattr(trainer.kw.common, "set_log_level", unexpected)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sampler = LocalSampler()
    saver = RecordingSaver()
    batches = [torch.tensor([[0., 1.], [1., 0.]])]
    model_trainer = trainer.Trainer(batches, saver, sampler, 2, 1, 1)

    model_trainer.train(max_steps=1, save_path="unused")

    assert sampler.widths == [4, 2, 2, 3, 3]
    assert len(saver.losses) == 1
    assert saver.losses[0][0] == 1
    assert np.isfinite(saver.losses[0][1])
    assert not batches[0].is_shared()
