"""Regression tests for QVAE MNIST FeatureExtractor device tracking."""

import os
import sys
import types
import unittest

import torch
from torch import nn

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

example_model_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../example/qvae_mnist/model")
)
sys.path.insert(0, example_model_root)

from feature_extractor import FeatureExtractor
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae import QVAE


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


class TestFeatureExtractorDevice(unittest.TestCase):
    """FeatureExtractor must follow parameter device after ``Module.to()``."""

    def setUp(self) -> None:
        self.input_dim = 8
        self.latent_dim = 4
        self.config = types.SimpleNamespace(
            num_latent_units=self.latent_dim,
            loss_type="mse",
            dist_beta=1.0,
            kl_beta=1e-6,
            weight_decay=0.01,
        )
        self.encoder = DummyEncoder(self.latent_dim)
        self.decoder = DummyDecoder(self.input_dim)
        self.bm = RestrictedBoltzmannMachine(2, 2)
        self.qvae = DummyQVAE(
            input_dimension=self.input_dim,
            activation_fct=None,
            config=self.config,
            encoder=self.encoder,
            decoder=self.decoder,
            bm=self.bm,
            sampler=None,
        )
        self.extractor = FeatureExtractor(self.qvae)

    def test_device_property_tracks_parameter_device_after_move(self):
        """``.to()`` on the QVAE must be reflected by ``FeatureExtractor.device``."""
        self.assertEqual(self.extractor.device, next(self.qvae.parameters()).device)

        self.qvae.to(torch.device("meta"))

        self.assertEqual(
            self.extractor.device,
            next(self.qvae.parameters()).device,
            "FeatureExtractor kept a stale device snapshot after the model moved",
        )

    def test_transform_sends_inputs_to_the_model_parameter_device(self):
        """Batched inference must place activations on the same device as the QVAE."""
        self.qvae.to(torch.device("meta"))
        batch = torch.zeros(2, self.input_dim)

        with self.assertRaises(NotImplementedError):
            # Meta tensors cannot be copied to NumPy; the forward pass still runs.
            self.extractor.transform(batch.numpy())

        self.assertEqual(self.encoder.last_input_device, torch.device("meta"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFeatureExtractorDeviceCuda(unittest.TestCase):
    """End-to-end check for construct-on-CUDA / move-to-CPU workflows."""

    def test_transform_after_cpu_move(self) -> None:
        input_dim = 8
        latent_dim = 4
        config = types.SimpleNamespace(
            num_latent_units=latent_dim,
            loss_type="mse",
            dist_beta=1.0,
            kl_beta=1e-6,
            weight_decay=0.01,
        )
        encoder = DummyEncoder(latent_dim)
        decoder = DummyDecoder(input_dim)
        bm = RestrictedBoltzmannMachine(2, 2)
        qvae = DummyQVAE(
            input_dimension=input_dim,
            activation_fct=None,
            config=config,
            encoder=encoder,
            decoder=decoder,
            bm=bm,
            sampler=None,
        )
        qvae.cuda()
        extractor = FeatureExtractor(qvae)
        qvae.cpu()

        features = extractor.transform(torch.zeros(2, input_dim).numpy())

        self.assertEqual(encoder.last_input_device.type, "cpu")
        self.assertEqual(features.shape, (2, latent_dim))


if __name__ == "__main__":
    unittest.main()
