"""Regression tests for CellQVAE encoder-parameter device alignment.

``CellQVAE`` overrides ``QVAE.forward`` and ``QVAE.energy`` and encodes through
``_encode``. A parent ``nn.Module._apply`` moves encoder parameters without
calling ``CellQVAE.to``. Caller batches that stay on CPU then mix with those
parameters inside ``QVAEEncoder``.

This file follows the merged #164 bar: meta-device mechanism tests on every
machine, plus CUDA construct / parent-move coverage behind ``skipUnless``.

Out of scope: ``qvae.py`` reconstruction and ``loss`` (parked
``cursor/qvae-recon-parameter-device-7c2a``), BM-facing ``QVAE.energy`` /
``bm_loss`` / ``_cross_entropy`` (#1348), and the qvae_cell MPS selector
(#512).
"""

# Parent moves go through ``_apply`` (see #164).
# pylint: disable=protected-access

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaiwu.torch_plugin import BoltzmannMachine  # pylint: disable=wrong-import-position


def _load_cell_models():
    """Load ``example/qvae_cell/models.py`` without making it a package."""
    path = ROOT / "example" / "qvae_cell" / "models.py"
    spec = importlib.util.spec_from_file_location("qvae_cell_models", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qvae_cell_models"] = module
    spec.loader.exec_module(module)
    return module


_CELL = _load_cell_models()
CellQVAE = _CELL.CellQVAE
QVAEEncoder = _CELL.QVAEEncoder
QVAEDecoder = _CELL.QVAEDecoder

_INPUT_DIM = 4
_HIDDEN_DIM = 6
_LATENT_DIM = 4


def _config(loss_type):
    """Build the small config ``CellQVAE`` reads during encode."""
    return SimpleNamespace(
        num_latent_units=_LATENT_DIM,
        loss_type=loss_type,
        dist_beta=1.0,
        kl_beta=0.0,
        weight_decay=0.0,
    )


def _make_model(loss_type="mse", n_batches=0, dtype=torch.float32):
    """Return a tiny ``CellQVAE`` whose submodules share ``dtype``."""
    encoder = QVAEEncoder(_INPUT_DIM, _HIDDEN_DIM, _LATENT_DIM, "layer")
    decoder = QVAEDecoder(
        _LATENT_DIM + n_batches,
        _HIDDEN_DIM,
        _INPUT_DIM,
        "layer",
    )
    model = CellQVAE(
        input_dimension=_INPUT_DIM,
        activation_fct=nn.ReLU(),
        config=_config(loss_type),
        encoder=encoder,
        decoder=decoder,
        bm=BoltzmannMachine(_LATENT_DIM, device="cpu"),
        sampler=object(),
        n_batches=n_batches,
    )
    model.to(dtype=dtype)
    model.eval()
    return model


class RecordingCellQVAE(CellQVAE):  # pylint: disable=too-few-public-methods
    """CellQVAE that records direct ``.to()`` calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_to_calls = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def _move_parameters_to_meta(module):
    """Move parameters the way a parent module does: ``_apply``, not ``.to``."""
    module._apply(lambda tensor: tensor.to(torch.device("meta")))


class TestCellQVAEEncoderParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_forward_after_apply_accepts_cpu_batch(self):
        """CPU expression matrices must encode once parameters live on meta."""
        model = _make_model()
        _move_parameters_to_meta(model)
        batch = torch.ones(2, _INPUT_DIM)

        recon, _, _, _ = model(batch)

        self.assertEqual(batch.device, torch.device("cpu"))
        self.assertEqual(recon.device, torch.device("meta"))
        self.assertEqual(recon.shape, (2, _INPUT_DIM))
        self.assertEqual(recon.dtype, torch.float32)

    def test_energy_after_apply_accepts_cpu_batch(self):
        """Energy uses the same encoder entry and must follow its parameters."""
        model = _make_model()
        _move_parameters_to_meta(model)
        batch = torch.zeros(2, _INPUT_DIM)

        energy = model.energy(batch, "mse")

        self.assertEqual(batch.device, torch.device("cpu"))
        self.assertEqual(energy.device, torch.device("meta"))
        self.assertEqual(energy.shape, (2,))

    def test_parent_apply_skips_child_to_and_still_encodes(self):
        """Parent ``_apply`` must not be required to call ``CellQVAE.to``."""
        encoder = QVAEEncoder(_INPUT_DIM, _HIDDEN_DIM, _LATENT_DIM, "layer")
        decoder = QVAEDecoder(_LATENT_DIM, _HIDDEN_DIM, _INPUT_DIM, "layer")
        model = RecordingCellQVAE(
            input_dimension=_INPUT_DIM,
            activation_fct=nn.ReLU(),
            config=_config("mse"),
            encoder=encoder,
            decoder=decoder,
            bm=BoltzmannMachine(_LATENT_DIM, device="cpu"),
            sampler=object(),
            n_batches=0,
        )
        model.eval()
        parent = nn.Module()
        parent.cell = model
        _move_parameters_to_meta(parent)

        recon, _, _, _ = parent.cell(torch.ones(2, _INPUT_DIM))

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(recon.device, torch.device("meta"))
        self.assertEqual(
            next(model.encoder.parameters()).device,
            torch.device("meta"),
        )

    def test_bernoulli_mean_follows_encoder_parameters(self):
        """Dataset-mean correction must happen on the encoder device."""
        model = _make_model(loss_type="bernoulli")
        model.set_dataset_mean(torch.full((_INPUT_DIM,), 0.25))
        model.set_train_bias(torch.full((_INPUT_DIM,), 0.25))
        _move_parameters_to_meta(model)
        batch = torch.full((2, _INPUT_DIM), 0.5)

        recon, _, _, _ = model(batch)

        self.assertEqual(recon.device, torch.device("meta"))
        self.assertEqual(recon.shape, (2, _INPUT_DIM))

    def test_batch_index_stays_with_cpu_caller_while_features_move(self):
        """Batch ids may stay on CPU; expression features still follow parameters."""
        model = _make_model(n_batches=2)
        _move_parameters_to_meta(model)
        batch = torch.ones(2, _INPUT_DIM)
        batch_idx = torch.tensor([0, 1])

        recon, _, _, _ = model(batch, batch_idx)

        self.assertEqual(batch_idx.device, torch.device("cpu"))
        self.assertEqual(recon.device, torch.device("meta"))
        self.assertEqual(recon.shape, (2, _INPUT_DIM))

    def test_float64_batch_keeps_dtype_on_parameter_device(self):
        """Device alignment must not downcast a float64 encoder batch."""
        model = _make_model(dtype=torch.float64)
        _move_parameters_to_meta(model)
        batch = torch.ones(2, _INPUT_DIM, dtype=torch.float64)

        recon, _, _, _ = model(batch)

        self.assertEqual(recon.dtype, torch.float64)
        self.assertEqual(recon.device, torch.device("meta"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCellQVAEEncoderDeviceAcrossAccelerators(unittest.TestCase):
    """Construct-on-CUDA and parent move-to-CPU coverage."""

    def test_cpu_batch_follows_cuda_encoder(self):
        """A CPU batch must score on CUDA after ``.cuda()``."""
        model = _make_model()
        model.cuda()

        recon, _, _, _ = model(torch.ones(2, _INPUT_DIM))
        energy = model.energy(torch.ones(2, _INPUT_DIM), "mse")

        self.assertEqual(recon.device.type, "cuda")
        self.assertEqual(energy.device.type, "cuda")

    def test_parent_move_to_cpu_accepts_cuda_batch(self):
        """``parent.to('cpu')`` must pull a CUDA batch onto the encoder."""
        model = _make_model()
        model.cuda()
        parent = nn.Module()
        parent.cell = model
        parent.to("cpu")
        batch = torch.ones(2, _INPUT_DIM, device="cuda")

        recon, _, _, _ = parent.cell(batch)

        self.assertEqual(batch.device.type, "cuda")
        self.assertEqual(recon.device.type, "cpu")
        self.assertEqual(next(model.encoder.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
