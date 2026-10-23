"""Regression tests for CellQVAE reconstruction device alignment.

``CellQVAE`` overrides ``QVAE.forward`` and does not call the base method.
Open #1370 aligns ``QVAE.forward`` in ``src/kaiwu/torch_plugin/qvae.py``.
The parked branch ``cursor/cell-qvae-encode-parameter-device-4e91`` aligns
``CellQVAE._encode`` only. Moving the decoder through ``_apply`` (the path
``nn.Module.to`` uses for children; merged #164) leaves encoder latents and
the Bernoulli ``_train_bias`` buffer on another device. ``forward`` has to
follow the decoder's live parameter device.

These tests keep the encoder on the caller device so the failure is inside
``forward``'s decoder call, not ``_encode``.
"""

# Parent moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access

import os
import sys
import types
import unittest

import torch
from torch import nn

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
CELL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "example", "qvae_cell")
)
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, CELL_ROOT)

from models import (  # pylint: disable=wrong-import-position
    CellQVAE,
    QVAEDecoder,
    QVAEEncoder,
)
from kaiwu.torch_plugin import (  # pylint: disable=wrong-import-position
    RestrictedBoltzmannMachine,
)


class RecordingDecoder(QVAEDecoder):
    """Decoder that records direct ``.to()`` calls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class ParameterFreeDecoder(nn.Module):
    """Decoder stand-in with no parameters."""

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        """Return ``zeta`` unchanged."""
        return zeta


def _build_cell_qvae(
    loss_type: str = "mse",
    n_batches: int = 0,
    decoder: nn.Module | None = None,
) -> CellQVAE:
    """Return a small CellQVAE whose decoder owns real parameters."""
    input_dim = 6
    latent_dim = 4
    hidden_dim = 5
    config = types.SimpleNamespace(
        num_latent_units=latent_dim,
        loss_type=loss_type,
        dist_beta=1.0,
        kl_beta=0.0,
        weight_decay=0.0,
        sampler_type="sa",
    )
    decoder_input = latent_dim + n_batches
    if decoder is None:
        decoder = RecordingDecoder(decoder_input, hidden_dim, input_dim)
    model = CellQVAE(
        input_dimension=input_dim,
        activation_fct=nn.ReLU(),
        config=config,
        encoder=QVAEEncoder(input_dim, hidden_dim, latent_dim),
        decoder=decoder,
        bm=RestrictedBoltzmannMachine(latent_dim // 2, latent_dim // 2),
        sampler=object(),
        n_batches=n_batches,
    )
    model.eval()
    return model


def _move_decoder(model: CellQVAE, device: torch.device) -> None:
    """Move only the decoder, the way a parent ``_apply`` moves a child."""
    parent = nn.Module()
    parent.decoder = model.decoder
    parent._apply(lambda tensor: tensor.to(device))


class TestCellQvaeForwardParameterDevice(unittest.TestCase):
    """Meta-device tests that run on CPU-only machines."""

    def test_decoder_parameter_device_falls_back_without_parameters(self):
        """A parameter-free decoder reports the fallback tensor device."""
        model = _build_cell_qvae(decoder=ParameterFreeDecoder())
        fallback = torch.zeros(1, device=torch.device("cpu"))

        self.assertEqual(
            model._decoder_parameter_device(fallback),
            torch.device("cpu"),
        )

        fallback = torch.zeros(1, device=torch.device("meta"))
        self.assertEqual(
            model._decoder_parameter_device(fallback),
            torch.device("meta"),
        )

    def test_parent_apply_does_not_call_decoder_to(self):
        """A parent move must not depend on the decoder ``.to()`` override."""
        model = _build_cell_qvae("mse")
        _move_decoder(model, torch.device("meta"))

        self.assertEqual(model.decoder.direct_to_calls, [])
        self.assertEqual(next(model.decoder.parameters()).device.type, "meta")
        self.assertEqual(next(model.encoder.parameters()).device.type, "cpu")

    def test_decoder_only_move_accepts_cpu_batch(self):
        """CPU latents must reconstruct on the decoder parameter device."""
        model = _build_cell_qvae("mse")
        _move_decoder(model, torch.device("meta"))
        features = torch.zeros(2, 6)

        recon_x, _, logits, zeta = model(features)

        self.assertEqual(features.device, torch.device("cpu"))
        self.assertEqual(logits.device.type, "cpu")
        self.assertEqual(zeta.device.type, "cpu")
        self.assertEqual(recon_x.device.type, "meta")
        self.assertEqual(recon_x.shape, (2, 6))

    def test_bernoulli_bias_follows_decoder_parameters(self):
        """A CPU ``_train_bias`` must not be added to meta reconstructions."""
        model = _build_cell_qvae("bernoulli")
        model.set_train_bias(0.25)
        _move_decoder(model, torch.device("meta"))
        features = torch.zeros(2, 6)

        recon_x, _, _, _ = model(features)

        self.assertEqual(model._train_bias.device.type, "cpu")
        self.assertEqual(recon_x.device.type, "meta")
        self.assertEqual(recon_x.shape, (2, 6))

    def test_batch_one_hot_follows_decoder_parameters(self):
        """Batch indices stay put; the one-hot must join the decoder device."""
        model = _build_cell_qvae("mse", n_batches=3)
        _move_decoder(model, torch.device("meta"))
        features = torch.zeros(2, 6)
        batch_idx = torch.tensor([0, 2])

        recon_x, _, _, zeta = model(features, batch_idx)

        self.assertEqual(batch_idx.device, torch.device("cpu"))
        self.assertEqual(zeta.device.type, "cpu")
        self.assertEqual(recon_x.device.type, "meta")
        self.assertEqual(recon_x.shape, (2, 6))

    def test_float64_batch_keeps_dtype_after_decoder_move(self):
        """A float64 batch must not be downcast when the decoder moves."""
        model = _build_cell_qvae("mse")
        model.double()
        _move_decoder(model, torch.device("meta"))
        features = torch.ones(2, 6, dtype=torch.float64)

        recon_x, _, logits, zeta = model(features)

        self.assertEqual(next(model.decoder.parameters()).dtype, torch.float64)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(zeta.dtype, torch.float64)
        self.assertEqual(recon_x.dtype, torch.float64)
        self.assertEqual(recon_x.device.type, "meta")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCellQvaeForwardParameterDeviceCuda(unittest.TestCase):
    """Construct-on-CPU / decoder-on-CUDA path."""

    def test_cpu_batch_matches_cpu_decoder_reference(self):
        """CPU batches must match a CPU decoder after the decoder moves to CUDA."""
        reference = _build_cell_qvae("bernoulli")
        reference.set_train_bias(0.25)
        model = _build_cell_qvae("bernoulli")
        model.load_state_dict(reference.state_dict())
        model.set_train_bias(0.25)
        _move_decoder(model, torch.device("cuda"))
        features = torch.rand(2, 6)

        torch.manual_seed(0)
        expected, _, _, _ = reference(features)
        torch.manual_seed(0)
        recon_x, _, _, _ = model(features)

        self.assertEqual(recon_x.device.type, "cuda")
        self.assertEqual(model._train_bias.device.type, "cpu")
        torch.testing.assert_close(recon_x.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
