"""Regression tests for QVAE MNIST ``FeatureExtractor`` device tracking.

The example ``FeatureExtractor`` routes batches through ``data.to(self.device)``
before calling the QVAE. Caching ``self.device`` at construction time leaves a
stale snapshot when the QVAE is moved with ``nn.Module.to()``—including moves
initiated by a parent wrapper module—so inference sends activations to the wrong
device while parameters already live on another. The fix derives ``device`` from
``next(model.parameters()).device`` on each access.

CUDA-dependent tests cover construct-on-CUDA / move-to-CPU workflows that
CPU-only machines cannot exercise.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

example_model_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../example/qvae_mnist/model")
)
sys.path.insert(0, example_model_root)

from feature_extractor import FeatureExtractor  # pylint: disable=wrong-import-position
from kaiwu.torch_plugin import (
    RestrictedBoltzmannMachine,
)  # pylint: disable=wrong-import-position
from kaiwu.torch_plugin.qvae import QVAE  # pylint: disable=wrong-import-position


class DummyEncoder(nn.Module):
    """Encoder that records the device of incoming activations."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.last_input_device: torch.device | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input_device = x.device
        return torch.ones(x.size(0), self.latent_dim, device=x.device, dtype=x.dtype)


class DummyDecoder(nn.Module):
    """Decoder that mirrors the latent batch on the latent device."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z.size(0), self.input_dim, device=z.device, dtype=z.dtype)


class DummyQVAE(QVAE):
    """Minimal QVAE wired for device-tracking tests."""

    def _create_encoder(self):
        return self.encoder

    def _create_decoder(self):
        return self.decoder

    def _create_bm(self):
        return self.bm

    def _create_sampler(self, sampler_type):
        del sampler_type
        return self.sampler


class _LegacyFeatureExtractor:
    """Pre-fix helper that snapshots ``device`` in ``__init__`` (stale-device repro)."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.device = next(model.parameters()).device


def _build_qvae_stack(
    input_dim: int = 8,
    latent_dim: int = 4,
) -> tuple[DummyEncoder, DummyDecoder, DummyQVAE]:
    config = types.SimpleNamespace(
        num_latent_units=latent_dim,
        loss_type="mse",
        dist_beta=1.0,
        kl_beta=1e-6,
        weight_decay=0.01,
    )
    encoder = DummyEncoder(latent_dim)
    decoder = DummyDecoder(input_dim)
    bm = RestrictedBoltzmannMachine(latent_dim // 2, latent_dim - latent_dim // 2)
    qvae = DummyQVAE(
        input_dimension=input_dim,
        activation_fct=None,
        config=config,
        encoder=encoder,
        decoder=decoder,
        bm=bm,
        sampler=None,
    )
    return encoder, decoder, qvae


class TestFeatureExtractorDevice(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def setUp(self) -> None:
        self.input_dim = 8
        self.encoder, _, self.qvae = _build_qvae_stack(self.input_dim, 4)
        self.extractor = FeatureExtractor(self.qvae)

    def test_stale_init_snapshot_diverges_after_module_move(self):
        """Caching device at construction is the exact pre-fix failure mode."""
        legacy = _LegacyFeatureExtractor(self.qvae)
        self.assertEqual(legacy.device, next(self.qvae.parameters()).device)

        self.qvae.to(torch.device("meta"))

        self.assertNotEqual(
            legacy.device,
            next(self.qvae.parameters()).device,
            "init-time snapshot should diverge from parameter device after .to()",
        )
        self.assertEqual(
            self.extractor.device,
            next(self.qvae.parameters()).device,
        )

    def test_device_property_tracks_parameter_device_after_direct_move(self):
        """A direct ``.to()`` on the QVAE must be reflected by ``.device``."""
        self.assertEqual(self.extractor.device, next(self.qvae.parameters()).device)

        self.qvae.to(torch.device("meta"))

        self.assertEqual(
            self.extractor.device,
            next(self.qvae.parameters()).device,
            "FeatureExtractor kept a stale device snapshot after the model moved",
        )

    def test_parent_move_updates_reported_device(self):
        """Parent-initiated ``.to()`` must not leave a stale extractor device."""
        parent = nn.Module()
        parent.qvae = self.qvae
        extractor = FeatureExtractor(parent.qvae)

        parent.to(torch.device("meta"))

        self.assertEqual(extractor.device, next(self.qvae.parameters()).device)

    def test_transform_sends_inputs_to_parameter_device(self):
        """``transform`` must place activations on the QVAE parameter device."""
        self.qvae.to(torch.device("meta"))
        batch = torch.zeros(2, self.input_dim)

        with self.assertRaises(NotImplementedError):
            # Meta tensors cannot be copied to NumPy; forward still runs first.
            self.extractor.transform(batch.numpy())

        self.assertEqual(self.encoder.last_input_device, torch.device("meta"))

    def test_extract_dataloader_uses_parameter_device(self):
        """``extract`` must move each batch to the QVAE parameter device."""
        self.qvae.to(torch.device("meta"))
        dataset = TensorDataset(torch.zeros(2, self.input_dim), torch.zeros(2))
        loader = DataLoader(dataset, batch_size=2)

        with self.assertRaises(NotImplementedError):
            self.extractor.extract(loader, return_labels=False)

        self.assertEqual(self.encoder.last_input_device, torch.device("meta"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFeatureExtractorDeviceCuda(unittest.TestCase):
    """End-to-end coverage of construct-on-CUDA / move-to-CPU paths."""

    def test_transform_after_direct_cpu_move(self) -> None:
        """Direct ``.cpu()`` after CUDA construction must not leave a stale device."""
        encoder, _, qvae = _build_qvae_stack()
        qvae.cuda()
        extractor = FeatureExtractor(qvae)
        qvae.cpu()

        features = extractor.transform(torch.zeros(2, 8).numpy())

        self.assertEqual(encoder.last_input_device.type, "cpu")
        self.assertEqual(features.shape, (2, 4))

    def test_transform_after_parent_cpu_move(self) -> None:
        """Parent ``.cpu()`` must keep extractor batches on the parameter device."""
        encoder, _, qvae = _build_qvae_stack()
        parent = nn.Module()
        parent.qvae = qvae
        parent.cuda()
        extractor = FeatureExtractor(parent.qvae)
        parent.cpu()

        features = extractor.transform(torch.zeros(2, 8).numpy())

        self.assertEqual(encoder.last_input_device.type, "cpu")
        self.assertEqual(features.shape, (2, 4))

    def test_stale_snapshot_would_mismatch_cuda_after_cpu_move(self) -> None:
        """Legacy init snapshot stays on CUDA while parameters moved to CPU."""
        _, _, qvae = _build_qvae_stack()
        qvae.cuda()
        legacy = _LegacyFeatureExtractor(qvae)
        qvae.cpu()

        self.assertEqual(legacy.device.type, "cuda")
        self.assertEqual(next(qvae.parameters()).device.type, "cpu")

        extractor = FeatureExtractor(qvae)
        self.assertEqual(extractor.device.type, "cpu")
        features = extractor.transform(torch.zeros(2, 8).numpy())
        self.assertEqual(features.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
