"""ModelTuner must report the average per-batch loss, not a sample-scaled one."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "example" / "qvae_mnist")
)

from trainer.model_tuner import ModelTuner  # noqa: E402  pylint: disable=wrong-import-position


class _Loader:
    def __init__(self, batches, dataset_size):
        self.batches = batches
        self.dataset = [None] * dataset_size

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        zeros = torch.zeros(inputs.size(0), 2)
        return zeros, None, zeros, zeros

    def loss(self, inputs, output, posterior):
        del inputs, output, posterior
        return self.scale * 0 + 1.0


def _tuner() -> ModelTuner:
    tuner = ModelTuner(SimpleNamespace(type="QVAE", weight_decay=0.0))
    model = _Model()
    tuner.register_model(model)
    tuner.register_optimiser(torch.optim.SGD(model.parameters(), lr=0.0))
    batches = [
        (torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)),
        (torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)),
    ]
    tuner.register_dataLoaders(_Loader(batches, 8), _Loader(batches, 8))
    return tuner


def test_train_loss_is_the_average_per_batch() -> None:
    tuner = _tuner()
    assert abs(tuner.train(0) - 1.0) < 1e-6


def test_test_loss_is_the_average_per_batch() -> None:
    tuner = _tuner()
    loss, _, _, _ = tuner.test()
    assert isinstance(loss, torch.Tensor)
    assert abs(float(loss) - 1.0) < 1e-6
