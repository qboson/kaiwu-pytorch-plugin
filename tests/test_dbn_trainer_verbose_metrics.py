"""Regression test for the verbose pretraining diagnostics.

Two printed values were wrong:

* ``Output shape after layer N`` printed the layer's *input* shape
  (``data_in`` is still the input until after the epoch loop), so a 64 -> 16
  layer reported ``torch.Size([400, 64])``;
* ``Iteration i+1, Average Loss`` divided the accumulated loss by the full
  loader length instead of the number of batches processed so far.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402

from dbn_trainer import DBNTrainer  # noqa: E402
from kaiwu.torch_plugin import UnsupervisedDBN  # noqa: E402


class _StubSampler:
    def solve(self, ising_matrix):
        return np.zeros((2, ising_matrix.shape[0]), dtype=np.float32)


def _run(capsys):
    torch.manual_seed(0)
    x = np.random.RandomState(0).rand(400, 64).astype(np.float32)
    dbn = UnsupervisedDBN([16]).create_rbm_layer(64)

    losses = iter([1.0, 2.0])

    def fake_train_batch(self, rbm, optimizer, batch_x):
        del self, rbm, optimizer, batch_x
        return torch.tensor(next(losses))

    trainer = DBNTrainer(
        n_epochs_rbm=1,
        batch_size=200,  # 400 samples -> exactly two batches
        verbose=True,
        shuffle=False,
        random_state=0,
    )
    trainer.sampler = _StubSampler()
    original = DBNTrainer._train_batch
    DBNTrainer._train_batch = fake_train_batch
    try:
        trainer.train(dbn, x)
    finally:
        DBNTrainer._train_batch = original
    return capsys.readouterr().out


def test_verbose_output_shape_is_the_layer_output(capsys):
    output = _run(capsys)

    assert "Output shape after layer 1: torch.Size([400, 16])" in output


def test_verbose_running_average_covers_seen_batches(capsys):
    output = _run(capsys)

    # first batch only -> the running average equals that batch's loss
    assert "Iteration 1, Average Loss: 1.000000" in output
