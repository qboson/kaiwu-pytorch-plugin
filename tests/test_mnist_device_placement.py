# pylint: disable=wrong-import-position,wrong-import-order,import-outside-toplevel
# pylint: disable=missing-function-docstring,too-few-public-methods,consider-using-with
# pylint: disable=no-member,redefined-outer-name,import-error
"""Regression tests for MNIST QVAE trainer device placement (#164 bar).

Refactored ``example/qvae_mnist`` never called ``model.to(...)`` and
``ModelTuner`` fed CPU batches straight into the QVAE. ``MnistQVAE._create_bm``
builds a Boltzmann machine that defaults to CUDA when available, while encoder and
decoder ``nn.Linear`` modules stay on CPU until an explicit placement step. The
first ``QVAE.loss`` → ``_cross_entropy`` → ``bm(q_prob)`` call then mixes
devices on CUDA hosts.

Mechanism tests run everywhere; CUDA ``@skipUnless`` cases cover the
construct-on-CUDA / trainer-unifies path that CPU-only machines cannot exercise.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

import kaiwu  # noqa: E402

kaiwu_src_path = os.path.join(src_root, "kaiwu")
if kaiwu_src_path not in kaiwu.__path__:
    kaiwu.__path__ = [kaiwu_src_path] + list(kaiwu.__path__)

_kaiwu_classical = types.ModuleType("kaiwu.classical")


class _StubSimulatedAnnealingOptimizer:
    def __init__(self, *args, **kwargs):
        pass

    def solve(self, ising_mat):
        num_spins = ising_mat.shape[0] - 1
        rng = np.random.default_rng(0)
        return rng.choice([-1.0, 1.0], size=(16, num_spins + 1))


_kaiwu_classical.SimulatedAnnealingOptimizer = _StubSimulatedAnnealingOptimizer
sys.modules["kaiwu.classical"] = _kaiwu_classical

_kaiwu_cim = types.ModuleType("kaiwu.cim")
_kaiwu_cim.CIMOptimizer = _StubSimulatedAnnealingOptimizer
_kaiwu_cim.PrecisionReducer = lambda optimizer, **kwargs: optimizer
sys.modules["kaiwu.cim"] = _kaiwu_cim

_kaiwu_common = types.ModuleType("kaiwu.common")
_kaiwu_common.CheckpointManager = types.SimpleNamespace(save_dir=".")
sys.modules["kaiwu.common"] = _kaiwu_common

_gif_stub = types.ModuleType("gif")
_gif_stub.frame = lambda fn: fn
_gif_stub.save = lambda *args, **kwargs: None
sys.modules["gif"] = _gif_stub

for _module in (
    "imageio",
    "torchvision",
    "torchmetrics",
    "pandas",
    "seaborn",
    "tqdm",
):
    pytest.importorskip(_module)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

_example_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../example/qvae_mnist")
)
sys.path.insert(0, _example_root)

from model.config import Config  # noqa: E402  pylint: disable=wrong-import-position
from model.model import MnistQVAE  # noqa: E402  pylint: disable=wrong-import-position
from trainer.model_tuner import (
    ModelTuner,
)  # noqa: E402  pylint: disable=wrong-import-position
from trainer.trainer import Trainer  # noqa: E402  pylint: disable=wrong-import-position


def _tiny_features(rows: int = 8, cols: int = 4) -> tuple[np.ndarray, np.ndarray]:
    features = np.random.RandomState(0).rand(rows, cols).astype(np.float32)
    labels = np.zeros(rows, dtype=np.int64)
    return features, labels


def _trainer(
    tmp_path,
    use_cuda: bool,
    *,
    rows: int = 8,
    cols: int = 4,
) -> Trainer:
    features, labels = _tiny_features(rows, cols)
    config = Config(
        "QVAE",
        output_dir=str(tmp_path),
        batch_size=4,
        num_train_samples=rows,
        num_test_samples=rows,
        use_cuda=use_cuda,
    )
    return Trainer(
        config,
        custom_train_data=(features, labels),
        custom_test_data=(features, labels),
    )


class TestTrainerDeviceResolution(unittest.TestCase):
    """Device flag resolution (runs on every machine)."""

    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_use_cuda_false_always_resolves_cpu(self):
        trainer = _trainer(self._tmpdir.name, use_cuda=False)
        self.assertEqual(trainer.device, torch.device("cpu"))

    def test_use_cuda_true_falls_back_when_cuda_unavailable(self):
        trainer = _trainer(self._tmpdir.name, use_cuda=True)
        if not torch.cuda.is_available():
            self.assertEqual(trainer.device.type, "cpu")
        else:
            self.assertEqual(trainer.device.type, "cuda")


class TestTrainerModelPlacement(unittest.TestCase):
    """Placement after ``Trainer._create_model`` (CPU CI)."""

    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_create_model_unifies_submodule_devices(self):
        trainer = _trainer(self._tmpdir.name, use_cuda=False)
        trainer._setup_data()  # pylint: disable=protected-access
        model = trainer._create_model()  # pylint: disable=protected-access

        devices = {parameter.device for parameter in model.parameters()}
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices.pop(), trainer.device)
        self.assertEqual(model.bm.device, trainer.device)

    def test_bm_matches_trainer_when_use_cuda_flag_is_true(self):
        trainer = _trainer(self._tmpdir.name, use_cuda=True)
        trainer._setup_data()  # pylint: disable=protected-access
        model = trainer._create_model()  # pylint: disable=protected-access

        device_types = {parameter.device.type for parameter in model.parameters()}
        self.assertEqual(device_types, {trainer.device.type})
        self.assertEqual(model.bm.device, trainer.device)


class TestModelTunerBatchPlacement(unittest.TestCase):
    """``ModelTuner`` must move DataLoader batches to the model device."""

    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        trainer = _trainer(self._tmpdir.name, use_cuda=False)
        trainer._setup_data()  # pylint: disable=protected-access
        self.model = trainer._create_model()  # pylint: disable=protected-access
        self.config = trainer.config
        features, labels = _tiny_features(8, 4)
        dataset = TensorDataset(
            torch.FloatTensor(features),
            torch.LongTensor(labels),
        )
        self.train_loader = DataLoader(dataset, batch_size=4)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_train_moves_batches_before_forward(self):
        seen_devices: list[torch.device] = []
        original_forward = self.model.forward

        def recording_forward(x):
            seen_devices.append(x.device)
            return original_forward(x)

        self.model.forward = recording_forward
        self.model.train()

        tuner = ModelTuner(config=self.config)
        tuner.register_model(self.model)
        tuner.register_dataLoaders(self.train_loader, self.train_loader)
        tuner.register_optimiser(
            torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        )

        tuner.train(epoch=1)

        model_device = next(self.model.parameters()).device
        self.assertTrue(seen_devices)
        self.assertTrue(all(device == model_device for device in seen_devices))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCudaTrainerPlacement(unittest.TestCase):
    """CUDA-only repro and placement (construct-on-CUDA / unify)."""

    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_create_networks_alone_splits_encoder_and_bm(self):
        """Anchor: without ``Trainer`` placement, Linear stays CPU and BM is CUDA."""
        config = Config("QVAE", output_dir=self._tmpdir.name)
        model = MnistQVAE(
            input_dimension=4,
            activation_fct=config.activation_fct,
            config=config,
        )
        model.create_networks()

        encoder_device = next(model.encoder.parameters()).device
        bm_device = model.bm.quadratic_coef.device

        self.assertEqual(encoder_device.type, "cpu")
        self.assertEqual(bm_device.type, "cuda")

    def test_trainer_unifies_when_use_cuda_false(self):
        trainer = _trainer(self._tmpdir.name, use_cuda=False)
        trainer._setup_data()  # pylint: disable=protected-access
        model = trainer._create_model()  # pylint: disable=protected-access

        self.assertEqual(trainer.device.type, "cpu")
        devices = {parameter.device.type for parameter in model.parameters()}
        self.assertEqual(devices, {"cpu"})

    def test_trainer_places_on_cuda_when_use_cuda_true(self):
        trainer = _trainer(self._tmpdir.name, use_cuda=True)
        trainer._setup_data()  # pylint: disable=protected-access
        model = trainer._create_model()  # pylint: disable=protected-access

        self.assertEqual(trainer.device.type, "cuda")
        devices = {parameter.device.type for parameter in model.parameters()}
        self.assertEqual(devices, {"cuda"})

    def test_single_train_step_runs_on_cuda(self):
        import model.model as mnist_model

        original_sampler = mnist_model.SimulatedAnnealingOptimizer
        mnist_model.SimulatedAnnealingOptimizer = _StubSimulatedAnnealingOptimizer
        try:
            features = np.random.RandomState(0).rand(8, 4).astype(np.float32)
            labels = np.zeros(8, dtype=np.int64)
            config = Config(
                "QVAE",
                output_dir=self._tmpdir.name,
                batch_size=4,
                num_epochs=1,
                use_cuda=True,
            )
            trainer = Trainer(
                config,
                custom_train_data=(features, labels),
                custom_test_data=(features, labels),
            )
            trainer._setup_data()  # pylint: disable=protected-access
            trainer._create_model()  # pylint: disable=protected-access
            trainer._setup_tuner()  # pylint: disable=protected-access
            loss = trainer.tuner.train(epoch=1)
            self.assertTrue(np.isfinite(loss))
        finally:
            mnist_model.SimulatedAnnealingOptimizer = original_sampler


def test_training_smoke_produces_checkpoint(tmp_path, monkeypatch):
    """End-to-end CPU smoke: finite losses and a saved checkpoint."""
    import model.model as mnist_model

    monkeypatch.setattr(
        mnist_model, "SimulatedAnnealingOptimizer", _StubSimulatedAnnealingOptimizer
    )
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


if __name__ == "__main__":
    unittest.main()
