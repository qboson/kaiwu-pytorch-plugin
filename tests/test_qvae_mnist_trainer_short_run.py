"""QVAE-MNIST Trainer must support short runs when logging snapshots."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "example" / "qvae_mnist")
)

from trainer.trainer import Trainer  # noqa: E402  pylint: disable=wrong-import-position


class _Tuner:  # pylint: disable=too-few-public-methods
    def train(self, epoch):
        del epoch
        return 0.5

    def test(self):
        inputs = torch.zeros(2, 4)
        return 0.5, inputs, torch.zeros(2, 4), [0, 1]

    def save_model(self, config_string="final"):
        del config_string


def test_short_training_run_does_not_divide_by_zero(tmp_path) -> None:
    config = SimpleNamespace(
        type="QVAE",
        num_epochs=1,
        output_dir=str(tmp_path),
        batch_size=2,
    )
    trainer = Trainer(config)
    trainer.tuner = _Tuner()
    trainer._setup_data = lambda: None  # pylint: disable=protected-access
    trainer._create_model = lambda: None  # pylint: disable=protected-access
    trainer._setup_tuner = lambda: None  # pylint: disable=protected-access
    trainer.test_loader = [(torch.zeros(2, 4), torch.tensor([0, 1]))]
    trainer._save_reconstruction = lambda *args: None  # pylint: disable=protected-access
    trainer._plot_training_curve = lambda: None  # pylint: disable=protected-access

    model, train_losses, test_losses = trainer.train()

    assert model is None
    assert train_losses == [0.5]
    assert test_losses == [0.5]
