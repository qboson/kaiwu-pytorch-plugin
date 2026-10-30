"""DBN trainer logging must report running averages and readable stats."""

import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "example" / "dbn_digits")
)

from dbn_trainer import DBNTrainer  # noqa: E402  pylint: disable=wrong-import-position
from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402  pylint: disable=wrong-import-position


class _StubSampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


def _trainer() -> DBNTrainer:
    trainer = DBNTrainer(
        use_cim=False,
        verbose=True,
        n_epochs_rbm=1,
        batch_size=2,
        shuffle=False,
        plot_img=False,
    )
    trainer.sampler = _StubSampler()
    trainer._train_batch = lambda rbm, optimizer, batch: torch.tensor(2.0)
    return trainer


def test_running_average_and_stats_format(capsys) -> None:
    trainer = _trainer()
    rbm = RestrictedBoltzmannMachine(num_visible=4, num_hidden=3)

    trainer._train_rbm_layer(  # pylint: disable=protected-access
        rbm, np.zeros((4, 4), dtype=np.float32), 0
    )
    output = capsys.readouterr().out

    assert "Iteration 1, Average Loss: 2.000000" in output
    assert re.search(r"jmean [\d.]+jmax", output) is None
    assert re.search(r"hmean [\d.]+hmax", output) is None
