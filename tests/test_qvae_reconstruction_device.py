"""Regression tests for QVAE reconstruction device alignment.

``QVAE.forward`` runs the encoder, the decoder, and the Bernoulli
``_train_bias`` buffer as three different pieces of module state. A parent
``_apply`` (the path ``nn.Module.to`` uses for children; merged #164) moves
whichever of those pieces actually own parameters or buffers, while a caller
batch often stays on CPU. Reconstruction has to follow the live parameter
device of the submodule it is about to touch.

Open #1348 aligns BM-facing latents in ``energy`` / ``bm_loss`` /
``_cross_entropy``. This file does not cover those paths.
"""

# Parent moves go through ``_apply`` (see #164).
# pylint: disable=protected-access

import types
import unittest
from unittest import mock

import torch
from torch import nn

from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae import QVAE


class LinearEncoder(nn.Module):
    """Encoder with one linear map, so the parameter device is observable."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
        self.seen_devices: list[torch.device] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Record the input device and return latent logits."""
        self.seen_devices.append(inputs.device)
        return self.fc(inputs)


class LinearDecoder(nn.Module):
    """Decoder with one linear map, so the parameter device is observable."""

    def __init__(self, latent_dim: int, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, input_dim)
        self.seen_devices: list[torch.device] = []

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        """Record the latent device and return reconstruction logits."""
        self.seen_devices.append(zeta.device)
        return self.fc(zeta)


class RecordingQVAE(QVAE):
    """QVAE that records direct ``.to()`` calls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def _create_encoder(self):
        """Return the encoder supplied to the constructor."""
        return self.encoder

    def _create_decoder(self):
        """Return the decoder supplied to the constructor."""
        return self.decoder

    def _create_bm(self):
        """Return the Boltzmann machine supplied to the constructor."""
        return self.bm

    def _create_sampler(self, sampler_type):
        """Sampling is not used by reconstruction tests."""
        del sampler_type

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def _build_qvae(loss_type: str = "mse") -> RecordingQVAE:
    """Return a small QVAE whose encoder and decoder own real parameters."""
    input_dim = 8
    latent_dim = 4
    config = types.SimpleNamespace(
        num_latent_units=latent_dim,
        loss_type=loss_type,
        dist_beta=1.0,
        kl_beta=0.0,
        weight_decay=0.0,
    )
    return RecordingQVAE(
        input_dimension=input_dim,
        activation_fct=None,
        config=config,
        encoder=LinearEncoder(input_dim, latent_dim),
        decoder=LinearDecoder(latent_dim, input_dim),
        bm=RestrictedBoltzmannMachine(latent_dim // 2, latent_dim // 2),
        sampler=object(),
    )


class TestQvaeReconstructionParameterDevice(unittest.TestCase):
    """Meta-device tests that run on CPU-only machines."""

    def test_parent_apply_move_accepts_cpu_batch(self):
        """CPU batches must reconstruct on the parameter device after ``_apply``."""
        parent = nn.Module()
        parent.qvae = _build_qvae("mse")
        parent.qvae.eval()
        parent._apply(lambda tensor: tensor.to(torch.device("meta")))

        recon_x, _, logits, _ = parent.qvae(torch.randn(2, 8))

        self.assertEqual(parent.qvae.direct_to_calls, [])
        self.assertEqual(parent.qvae.encoder.seen_devices[-1].type, "meta")
        self.assertEqual(parent.qvae.decoder.seen_devices[-1].type, "meta")
        self.assertEqual(logits.device.type, "meta")
        self.assertEqual(recon_x.device.type, "meta")
        self.assertEqual(recon_x.shape, (2, 8))

    def test_decoder_only_move_aligns_latents_and_train_bias(self):
        """Moving only the decoder must not add a CPU bias to meta logits."""
        qvae = _build_qvae("bernoulli")
        qvae.eval()
        qvae.set_train_bias(0.25)
        qvae.decoder._apply(lambda tensor: tensor.to(torch.device("meta")))

        recon_x, _, logits, _ = qvae(torch.zeros(2, 8))

        self.assertEqual(qvae._train_bias.device.type, "cpu")
        self.assertEqual(logits.device.type, "cpu")
        self.assertEqual(qvae.decoder.seen_devices[-1].type, "meta")
        self.assertEqual(recon_x.device.type, "meta")
        self.assertEqual(recon_x.shape, (2, 8))

    def test_loss_aligns_cpu_targets_with_decoder_parameters(self):
        """``loss`` must score targets on the decoder parameter device."""
        qvae = _build_qvae("mse")
        qvae.eval()
        qvae.decoder._apply(lambda tensor: tensor.to(torch.device("meta")))
        recon_x = torch.zeros(2, 8, device=torch.device("meta"))
        kl_term = torch.zeros(2, device=torch.device("meta"))
        weight_decay = torch.zeros((), device=torch.device("meta"))

        with mock.patch.object(qvae, "_kl_dist_from", return_value=kl_term):
            with mock.patch.object(
                qvae, "_weight_decay_loss", return_value=weight_decay
            ):
                loss = qvae.loss(torch.ones(2, 8), recon_x, posterior=None)

        self.assertEqual(loss.device.type, "meta")
        self.assertEqual(loss.shape, ())

    def test_float64_batch_keeps_dtype_after_apply_move(self):
        """A float64 batch must not be downcast when parameters move to meta."""
        qvae = _build_qvae("mse")
        qvae.double()
        qvae.eval()
        qvae._apply(lambda tensor: tensor.to(torch.device("meta")))

        recon_x, _, logits, _ = qvae(torch.ones(2, 8, dtype=torch.float64))

        self.assertEqual(qvae.encoder.fc.weight.dtype, torch.float64)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(recon_x.dtype, torch.float64)
        self.assertEqual(recon_x.device.type, "meta")

    def test_cpu_forward_still_matches_shapes(self):
        """An already-aligned CPU batch keeps the public reconstruction contract."""
        qvae = _build_qvae("mse")
        qvae.eval()
        inputs = torch.ones(3, 8)

        recon_x, posterior, logits, zeta = qvae(inputs)

        self.assertEqual(recon_x.device, torch.device("cpu"))
        self.assertEqual(logits.shape, (3, 4))
        self.assertEqual(zeta.shape, (3, 4))
        self.assertEqual(posterior.logit_mu.shape, logits.shape)
        self.assertEqual(recon_x.shape, inputs.shape)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestQvaeReconstructionParameterDeviceCuda(unittest.TestCase):
    """Accelerator coverage for parent moves between CPU and CUDA."""

    def test_cpu_batch_after_parent_move_to_cuda(self):
        """A CPU batch must reconstruct after ``parent.to('cuda')``."""
        parent = nn.Module()
        parent.qvae = _build_qvae("mse")
        parent.qvae.eval()
        parent.to("cuda")

        recon_x, _, logits, _ = parent.qvae(torch.randn(2, 8))

        self.assertEqual(parent.qvae.direct_to_calls, [])
        self.assertEqual(recon_x.device.type, "cuda")
        self.assertEqual(logits.device.type, "cuda")

    def test_cuda_batch_after_parent_move_back_to_cpu(self):
        """Parameters moved CUDA then CPU must accept a CUDA batch."""
        parent = nn.Module()
        parent.qvae = _build_qvae("bernoulli")
        parent.qvae.eval()
        parent.qvae.set_train_bias(0.25)
        parent.to("cuda")
        parent.to("cpu")

        recon_x, _, _, _ = parent.qvae(torch.zeros(2, 8, device="cuda"))

        self.assertEqual(next(parent.qvae.encoder.parameters()).device.type, "cpu")
        self.assertEqual(recon_x.device.type, "cpu")
        self.assertEqual(recon_x.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
