"""Regression for conditional BM training with visible outputs and hidden nodes."""

from pathlib import Path

import numpy as np
import pytest
import torch


class _FixedSampler:
    def solve(self, ising_matrix):
        # The final spin is the gauge. The hidden states differ between the
        # full-visible and input-only conditional samples.
        spins = {
            3: [-1, 1, 1],
            4: [-1, -1, -1, 1],
            6: [-1, -1, -1, -1, -1, 1],
        }
        return np.array([spins[ising_matrix.shape[0]]])


class _InlinePool:
    def map(self, worker, batches):
        return list(map(worker, batches))

    def close(self):
        pass

    def join(self):
        pass


class _RecordingSaver:
    def save_info(self, *args):
        pass

    def output_loss(self, step, kl_divergence, ncl, total):
        self.ncl = ncl


@pytest.mark.parametrize(("num_output", "expected_ncl"), [(1, -2.0), (0, 0.0)])
def test_ncl_masks_visible_outputs_without_masking_hidden_nodes(
    monkeypatch, tmp_path, num_output, expected_ncl
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from example.bm_generation import trainer

    monkeypatch.setattr(trainer.mp, "Pool", lambda processes: _InlinePool())
    monkeypatch.setattr(trainer.kw.common, "set_log_level", lambda level: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    saver = _RecordingSaver()
    model_trainer = trainer.Trainer(
        [torch.tensor([[0.0, 1.0, 1.0]])],
        saver,
        _FixedSampler(),
        num_visible=3,
        num_hidden=2,
        num_output=num_output,
    )
    with torch.no_grad():
        model_trainer.bm_net.quadratic_coef.zero_()
        model_trainer.bm_net.linear_bias.copy_(torch.tensor([0.0, 0.0, 7.0, 0.0, 2.0]))

    model_trainer.train(max_steps=1, save_path=tmp_path)

    # Once the visible output is zeroed, only the second hidden node differs.
    # Its linear bias of 2 gives an energy difference of -2. Without visible outputs,
    # the example keeps the NCL term at zero.
    assert saver.ncl == pytest.approx(expected_ncl)
