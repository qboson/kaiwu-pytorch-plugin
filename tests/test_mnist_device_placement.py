"""Regression test for device placement in the MNIST QVAE trainer."""

import os
import sys

import numpy as np
import pytest

for _module in ("gif", "imageio", "torchvision", "torchmetrics", "pandas", "seaborn", "tqdm"):
    pytest.importorskip(_module)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/qvae_mnist"))
)
import kaiwu.classical as kaiwu_classical  # noqa: E402  pylint: disable=wrong-import-position
from model.config import Config  # noqa: E402  pylint: disable=wrong-import-position
from trainer.trainer import Trainer  # noqa: E402  pylint: disable=wrong-import-position


class _StubSampler:
    """Legal Ising solutions without the license-gated SDK solver."""

    def __init__(self, *args, **kwargs):
        pass

    def solve(self, ising_mat):
        num_spins = ising_mat.shape[0] - 1
        rng = np.random.default_rng(0)
        return rng.choice([-1.0, 1.0], size=(16, num_spins + 1))



def _trainer(tmp_path, use_cuda):
    features = np.random.RandomState(0).rand(8, 4).astype(np.float32)
    labels = np.zeros(8, dtype=np.int64)
    config = Config(
        "QVAE",
        output_dir=str(tmp_path),
        batch_size=4,
        num_train_samples=8,
        num_test_samples=8,
        use_cuda=use_cuda,
    )
    return Trainer(
        config,
        custom_train_data=(features, labels),
        custom_test_data=(features, labels),
    )


def test_trainer_places_model_on_the_resolved_device(tmp_path):
    """use_cuda selects one device; the model must live on it."""
    trainer = _trainer(tmp_path, use_cuda=True)

    # CUDA is unavailable in the test environment, so the flag must resolve to CPU.
    assert trainer.device.type == "cpu"

    trainer._setup_data()  # pylint: disable=protected-access
    model = trainer._create_model()  # pylint: disable=protected-access

    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
    assert model.bm.device == trainer.device


def test_training_runs_with_the_placed_model(tmp_path, monkeypatch):
    """End-to-end smoke test: data and model must stay on the same device."""
    import model.model as mnist_model

    monkeypatch.setattr(mnist_model, "SimulatedAnnealingOptimizer", _StubSampler)
    features = np.random.RandomState(0).rand(16, 784).astype(np.float32)
    labels = np.random.RandomState(1).randint(0, 10, size=16)
    config = Config(
        "QVAE",
        output_dir=str(tmp_path),
        batch_size=16,
        num_epochs=10,
        num_train_samples=16,
        num_test_samples=16,
    )
    trainer = Trainer(
        config,
        custom_train_data=(features, labels),
        custom_test_data=(features, labels),
    )

    model, train_losses, test_losses = trainer.train()

    assert len(train_losses) == 10
    assert len(test_losses) == 10
    assert all(np.isfinite(value) for value in train_losses)
    assert (tmp_path / "model_final_QVAE.pt").is_file()
    assert trainer.device == next(model.parameters()).device
