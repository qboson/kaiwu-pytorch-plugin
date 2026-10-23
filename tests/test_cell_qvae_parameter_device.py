"""Regression tests for CellQVAE encoder-parameter device alignment.

``CellQVAE`` overrides ``QVAE.forward`` and ``QVAE.energy`` and encodes through
``_encode``. It does not call those base methods, so the library edits in
``qvae.py`` (#1370 reconstruction, #1348 BM-facing latents) do not cover this
path. A parent ``nn.Module._apply`` moves encoder parameters without calling
``CellQVAE.to`` or ``QVAEEncoder.to`` (merged #164). ``_encode`` rereads the
live encoder parameter device on every call. A parameter-free encoder falls
back to the batch device. Caller tensors are not modified in place.

This file follows ``tests/test_bm_device_sync.py``: meta-device mechanism
tests on every machine, plus CUDA parent-move coverage behind ``skipUnless``.

Out of scope: decoder-only reconstruction (sibling
``cursor/cell-qvae-forward-parameter-device-eb17``), the ``qvae.py`` hunks
themselves, and the qvae_cell MPS selector (#512). Tests that move only the
encoder check ``_encode``. They do not claim ``forward`` succeeds when the
decoder stays on another device.
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
_CPU = torch.device("cpu")
_META = torch.device("meta")


def _config(loss_type):
    """Build the small config ``CellQVAE`` reads during encode."""
    return SimpleNamespace(
        num_latent_units=_LATENT_DIM,
        loss_type=loss_type,
        dist_beta=1.0,
        kl_beta=0.0,
        weight_decay=0.0,
    )


def _make_model(
    loss_type="mse",
    n_batches=0,
    dtype=torch.float32,
    encoder=None,
    cell_cls=None,
):
    """Return a tiny ``CellQVAE`` whose submodules share ``dtype``."""
    if encoder is None:
        encoder = QVAEEncoder(_INPUT_DIM, _HIDDEN_DIM, _LATENT_DIM, "layer")
    decoder = QVAEDecoder(
        _LATENT_DIM + n_batches,
        _HIDDEN_DIM,
        _INPUT_DIM,
        "layer",
    )
    if cell_cls is None:
        cell_cls = CellQVAE
    model = cell_cls(
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


class RecordingEncoder(QVAEEncoder):
    """Encoder that records inputs and direct ``.to()`` calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_batches = []
        self.direct_to_calls = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)

    def forward(self, inputs):
        """Record ``inputs``, then encode."""
        self.seen_batches.append(inputs.detach())
        return super().forward(inputs)


class RecordingCellQVAE(CellQVAE):  # pylint: disable=too-few-public-methods
    """CellQVAE that records direct ``.to()`` calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_to_calls = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class ParameterFreeEncoder(nn.Module):  # pylint: disable=too-few-public-methods
    """Encoder stand-in with no parameters."""

    def forward(self, inputs):
        """Return ``inputs``. The fixture latent size matches the input size."""
        return inputs


def _make_recording_model(loss_type="mse", n_batches=0):
    """Return a model that records ``CellQVAE.to`` and ``encoder.to``."""
    encoder = RecordingEncoder(_INPUT_DIM, _HIDDEN_DIM, _LATENT_DIM, "layer")
    model = _make_model(
        loss_type,
        n_batches,
        encoder=encoder,
        cell_cls=RecordingCellQVAE,
    )
    # ``model.to(dtype=)`` during construction is not the parent path.
    model.direct_to_calls.clear()
    return model


def _move_module_to(module, device):
    """Move parameters the way a parent does: ``_apply``, not ``.to``."""
    module._apply(lambda tensor: tensor.to(device))


def _move_parameters_to_meta(module):
    """Move parameters onto ``meta`` without calling ``.to``."""
    _move_module_to(module, _META)


def _retarget_storage(device):
    """Return an ``_apply`` callback that places empty storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    """

    def _apply_fn(tensor):
        if tensor.device == device:
            return tensor
        return torch.empty(tensor.shape, dtype=tensor.dtype, device=device)

    return _apply_fn


def _cpu_encoder_logits(model, batch):
    """Return encoder logits for ``batch`` on a CPU copy of the weights."""
    state = {
        key: value.detach().to(device=_CPU).clone()
        for key, value in model.encoder.state_dict().items()
    }
    encoder = QVAEEncoder(_INPUT_DIM, _HIDDEN_DIM, _LATENT_DIM, "layer")
    weight = next(iter(state.values()))
    encoder.to(dtype=weight.dtype)
    encoder.load_state_dict(state)
    encoder.eval()
    features = batch.detach().to(dtype=weight.dtype, device=_CPU)
    if model.config.loss_type == "bernoulli" and model._dataset_mean is not None:
        features = features - torch.as_tensor(
            model._dataset_mean,
            dtype=features.dtype,
            device=features.device,
        )
    with torch.no_grad():
        return encoder(features)


def _reject_qvae_forward(*args, **kwargs):
    """Fail when ``CellQVAE.forward`` delegates to ``QVAE.forward``."""
    del args, kwargs
    raise AssertionError("CellQVAE.forward called QVAE.forward")


def _reject_qvae_energy(*args, **kwargs):
    """Fail when ``CellQVAE.energy`` delegates to ``QVAE.energy``."""
    del args, kwargs
    raise AssertionError("CellQVAE.energy called QVAE.energy")


class TestCellQVAEEncoderParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_forward_after_apply_accepts_cpu_batch(self):
        """CPU expression matrices must encode once parameters live on meta."""
        model = _make_model()
        _move_parameters_to_meta(model)
        batch = torch.ones(2, _INPUT_DIM)
        before = batch.clone()
        pointer = batch.data_ptr()

        recon, _, _, _ = model(batch)

        self.assertEqual(batch.device, _CPU)
        self.assertEqual(batch.data_ptr(), pointer)
        torch.testing.assert_close(batch, before)
        self.assertEqual(recon.device, _META)
        self.assertEqual(recon.shape, (2, _INPUT_DIM))
        self.assertEqual(recon.dtype, torch.float32)

    def test_energy_after_apply_accepts_cpu_batch(self):
        """Energy uses the same encoder entry and must follow its parameters."""
        model = _make_model()
        _move_parameters_to_meta(model)
        batch = torch.zeros(2, _INPUT_DIM)
        before = batch.clone()

        energy = model.energy(batch, "mse")

        self.assertEqual(batch.device, _CPU)
        torch.testing.assert_close(batch, before)
        self.assertEqual(energy.device, _META)
        self.assertEqual(energy.shape, (2,))

    def test_parent_apply_skips_child_to_and_still_encodes(self):
        """Parent ``_apply`` must not be required to call ``.to``."""
        model = _make_recording_model()
        parent = nn.Module()
        parent.cell = model
        _move_parameters_to_meta(parent)

        recon, _, logits, _ = parent.cell(torch.ones(2, _INPUT_DIM))

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(model.encoder.direct_to_calls, [])
        self.assertEqual(recon.device, _META)
        self.assertEqual(logits.device, _META)
        self.assertEqual(next(model.encoder.parameters()).device, _META)

    def test_bernoulli_mean_follows_encoder_parameters(self):
        """A CPU dataset mean is subtracted on the encoder parameter device."""
        encoder = RecordingEncoder(_INPUT_DIM, _HIDDEN_DIM, _LATENT_DIM, "layer")
        model = _make_model(loss_type="bernoulli", encoder=encoder)
        mean = torch.full((_INPUT_DIM,), 0.25)
        model.set_dataset_mean(mean)
        model.set_train_bias(torch.full((_INPUT_DIM,), 0.25))
        batch = torch.full((2, _INPUT_DIM), 0.5)
        mean_before = mean.clone()

        recon, _, _, _ = model(batch)

        torch.testing.assert_close(encoder.seen_batches[-1], batch - mean)
        torch.testing.assert_close(mean, mean_before)
        self.assertEqual(mean.device, _CPU)
        self.assertEqual(recon.device, _CPU)

        _move_parameters_to_meta(model)
        recon, _, _, _ = model(batch)

        self.assertEqual(encoder.direct_to_calls, [])
        self.assertEqual(batch.device, _CPU)
        self.assertEqual(mean.device, _CPU)
        self.assertEqual(encoder.seen_batches[-1].device, _META)
        self.assertEqual(recon.device, _META)
        self.assertEqual(recon.shape, (2, _INPUT_DIM))

    def test_batch_index_stays_with_cpu_caller_while_features_move(self):
        """Batch ids may stay on CPU; expression features still follow parameters."""
        model = _make_model(n_batches=2)
        _move_parameters_to_meta(model)
        batch = torch.ones(2, _INPUT_DIM)
        batch_idx = torch.tensor([0, 1])

        recon, _, _, _ = model(batch, batch_idx)

        self.assertEqual(batch_idx.device, _CPU)
        self.assertEqual(recon.device, _META)
        self.assertEqual(recon.shape, (2, _INPUT_DIM))

    def test_float64_batch_keeps_dtype_on_parameter_device(self):
        """Device alignment must not downcast a float64 encoder batch."""
        model = _make_model(dtype=torch.float64)
        _move_parameters_to_meta(model)
        batch = torch.ones(2, _INPUT_DIM, dtype=torch.float64)

        recon, _, logits, _ = model(batch)

        self.assertEqual(recon.dtype, torch.float64)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(recon.device, _META)

    def test_dtype_only_move_keeps_cpu_device(self):
        """``.to(dtype=...)`` must keep CPU and must not change logits."""
        model = _make_model()
        model.to(dtype=torch.float64)
        batch = torch.ones(2, _INPUT_DIM, dtype=torch.float64)

        recon, _, logits, _ = model(batch)

        self.assertEqual(next(model.encoder.parameters()).dtype, torch.float64)
        self.assertEqual(next(model.encoder.parameters()).device, _CPU)
        self.assertEqual(recon.dtype, torch.float64)
        self.assertEqual(recon.device, _CPU)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(logits.device, _CPU)
        torch.testing.assert_close(logits, _cpu_encoder_logits(model, batch))

    def test_parameter_device_is_reread_after_later_moves(self):
        """The helper must be reread after each move, not cached on first use."""
        model = _make_model()
        batch = torch.randn(2, _INPUT_DIM)

        self.assertEqual(model._encoder_parameter_device(batch), _CPU)
        _, _, first, _ = model(batch)
        self.assertEqual(first.device, _CPU)

        _move_parameters_to_meta(model)
        self.assertEqual(model._encoder_parameter_device(batch), _META)
        _, _, second, _ = model(batch)
        self.assertEqual(batch.device, _CPU)
        self.assertEqual(second.device, _META)

        # ``.to(cpu)`` cannot copy out of meta; ``_apply`` still retargets.
        model._apply(_retarget_storage(_CPU))
        self.assertEqual(model._encoder_parameter_device(batch), _CPU)
        _, _, third, _ = model(batch)
        energy = model.energy(batch, "mse")
        self.assertEqual(third.device, _CPU)
        self.assertEqual(energy.device, _CPU)
        self.assertEqual(energy.shape, (2,))
        self.assertEqual(batch.device, _CPU)

    def test_encoder_only_apply_accepts_cpu_batch(self):
        """Moving only the encoder must align ``_encode`` and not call ``.to``."""
        encoder = RecordingEncoder(_INPUT_DIM, _HIDDEN_DIM, _LATENT_DIM, "layer")
        model = _make_model(encoder=encoder)
        _move_module_to(model.encoder, _META)
        batch = torch.ones(2, _INPUT_DIM)
        before = batch.clone()

        logits, _, zeta = model._encode(batch)

        self.assertEqual(encoder.direct_to_calls, [])
        self.assertEqual(batch.device, _CPU)
        torch.testing.assert_close(batch, before)
        self.assertEqual(next(model.decoder.parameters()).device, _CPU)
        self.assertEqual(encoder.seen_batches[-1].device, _META)
        self.assertEqual(logits.device, _META)
        self.assertEqual(zeta.device, _META)

    def test_decoder_parameters_do_not_choose_the_encoder_device(self):
        """A meta decoder must not pull CPU encoder inputs off CPU."""
        encoder = RecordingEncoder(_INPUT_DIM, _HIDDEN_DIM, _LATENT_DIM, "layer")
        model = _make_model(encoder=encoder)
        _move_module_to(model.decoder, _META)
        batch = torch.randn(2, _INPUT_DIM)

        logits, _, _ = model._encode(batch)

        self.assertEqual(model._encoder_parameter_device(batch), _CPU)
        self.assertEqual(next(model.decoder.parameters()).device, _META)
        self.assertEqual(next(model.encoder.parameters()).device, _CPU)
        self.assertEqual(batch.device, _CPU)
        self.assertEqual(encoder.seen_batches[-1].device, _CPU)
        self.assertEqual(logits.device, _CPU)

    def test_parameter_free_encoder_keeps_the_batch_device(self):
        """With no encoder parameters, the batch device is the fallback."""
        model = _make_model(encoder=ParameterFreeEncoder())
        batch = torch.randn(2, _INPUT_DIM)

        self.assertEqual(list(model.encoder.parameters()), [])
        self.assertEqual(model._encoder_parameter_device(batch), _CPU)
        recon, _, logits, _ = model(batch)
        self.assertEqual(batch.device, _CPU)
        self.assertEqual(logits.device, _CPU)
        self.assertEqual(recon.device, _CPU)
        self.assertEqual(recon.shape, (2, _INPUT_DIM))

        meta_batch = torch.ones(2, _INPUT_DIM, device=_META)
        self.assertEqual(model._encoder_parameter_device(meta_batch), _META)
        meta_logits, _, _ = model._encode(meta_batch)
        self.assertEqual(meta_logits.device, _META)
        self.assertEqual(meta_batch.device, _META)

    def test_same_device_logits_match_the_encoder(self):
        """A CPU batch already on the parameter device must be unchanged."""
        model = _make_model()
        batch = torch.randn(2, _INPUT_DIM)
        before = batch.clone()
        pointer = batch.data_ptr()

        _, _, logits, _ = model(batch)

        torch.testing.assert_close(logits, _cpu_encoder_logits(model, batch))
        torch.testing.assert_close(batch, before)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(logits.device, _CPU)
        self.assertEqual(batch.device, _CPU)

    def test_same_device_alignment_keeps_autograd(self):
        """``.to`` on an already-aligned batch must not detach the graph."""
        model = _make_model()
        model.train()
        model.encoder.dropout.p = 0
        batch = torch.randn(2, _INPUT_DIM, requires_grad=True)

        _, _, logits, _ = model(batch)
        logits.sum().backward()

        weight = model.encoder.fc1.weight
        self.assertIsNotNone(batch.grad)
        self.assertEqual(batch.grad.device, _CPU)
        self.assertEqual(batch.grad.shape, batch.shape)
        self.assertTrue(torch.isfinite(batch.grad).all().item())
        self.assertIsNotNone(weight.grad)
        self.assertEqual(weight.grad.device, _CPU)
        self.assertTrue(torch.isfinite(weight.grad).all().item())
        self.assertGreater(float(weight.grad.abs().sum()), 0.0)

    def test_forward_does_not_call_qvae_forward(self):
        """``CellQVAE.forward`` must not depend on ``QVAE.forward`` (#1370)."""
        model = _make_model()
        batch = torch.ones(2, _INPUT_DIM)

        with unittest.mock.patch.object(_CELL.QVAE, "forward", _reject_qvae_forward):
            recon, _, _, _ = model(batch)

        self.assertEqual(recon.shape, (2, _INPUT_DIM))
        self.assertEqual(recon.device, _CPU)

    def test_energy_does_not_call_qvae_energy(self):
        """``CellQVAE.energy`` must not depend on ``QVAE.energy`` (#1348)."""
        model = _make_model()
        batch = torch.zeros(2, _INPUT_DIM)

        with unittest.mock.patch.object(_CELL.QVAE, "energy", _reject_qvae_energy):
            energy = model.energy(batch, "mse")

        self.assertEqual(energy.shape, (2,))
        self.assertEqual(energy.device, _CPU)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCellQVAEEncoderDeviceAcrossAccelerators(unittest.TestCase):
    """Construct-on-CUDA and parent move-to-CPU coverage."""

    def test_cpu_batch_follows_cuda_encoder(self):
        """A CPU batch must score on CUDA after ``.cuda()`` and match CPU weights."""
        model = _make_model()
        model.cuda()
        batch = torch.ones(2, _INPUT_DIM)
        before = batch.clone()

        recon, _, logits, _ = model(batch)
        energy = model.energy(batch, "mse")

        self.assertEqual(batch.device.type, "cpu")
        torch.testing.assert_close(batch, before)
        self.assertEqual(recon.device.type, "cuda")
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(energy.device.type, "cuda")
        torch.testing.assert_close(
            logits.detach().cpu(), _cpu_encoder_logits(model, batch)
        )

    def test_parent_move_to_cpu_accepts_cuda_batch(self):
        """``parent.to('cpu')`` must pull a CUDA batch onto the encoder."""
        model = _make_recording_model()
        model.cuda()
        recorded = list(model.direct_to_calls)
        parent = nn.Module()
        parent.cell = model
        parent.to("cpu")
        batch = torch.ones(2, _INPUT_DIM, device="cuda")

        recon, _, logits, _ = parent.cell(batch)

        self.assertEqual(model.direct_to_calls, recorded)
        self.assertEqual(model.encoder.direct_to_calls, [])
        self.assertEqual(batch.device.type, "cuda")
        self.assertEqual(recon.device.type, "cpu")
        self.assertEqual(logits.device.type, "cpu")
        self.assertEqual(next(model.encoder.parameters()).device.type, "cpu")
        torch.testing.assert_close(logits, _cpu_encoder_logits(model, batch))

    def test_parent_cuda_matches_cpu_encoder_logits(self):
        """``parent.to('cuda')`` accepts a CPU batch and does not call ``.to``."""
        model = _make_recording_model()
        parent = nn.Module()
        parent.cell = model
        parent.to("cuda")
        batch = torch.randn(2, _INPUT_DIM)
        before = batch.clone()

        recon, _, logits, _ = parent.cell(batch)

        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(model.encoder.direct_to_calls, [])
        self.assertEqual(batch.device.type, "cpu")
        torch.testing.assert_close(batch, before)
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(recon.device.type, "cuda")
        torch.testing.assert_close(
            logits.detach().cpu(), _cpu_encoder_logits(model, batch)
        )

    def test_backward_from_cpu_batch_reaches_cuda_encoder(self):
        """The device copy inside ``_encode`` must stay in the autograd graph."""
        model = _make_model()
        model.train()
        model.encoder.dropout.p = 0
        model.cuda()
        batch = torch.randn(2, _INPUT_DIM, requires_grad=True)

        _, _, logits, _ = model(batch)
        logits.sum().backward()

        weight = model.encoder.fc1.weight
        self.assertEqual(logits.device.type, "cuda")
        self.assertIsNotNone(batch.grad)
        self.assertEqual(batch.grad.device.type, "cpu")
        self.assertEqual(batch.grad.shape, batch.shape)
        self.assertTrue(torch.isfinite(batch.grad).all().item())
        self.assertIsNotNone(weight.grad)
        self.assertEqual(weight.grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(weight.grad).all().item())
        self.assertGreater(float(weight.grad.abs().sum()), 0.0)

    def test_parameter_device_follows_later_parent_moves(self):
        """Parent moves between CUDA and CPU must be reread, not cached."""
        model = _make_recording_model()
        parent = nn.Module()
        parent.cell = model
        cpu_batch = torch.randn(2, _INPUT_DIM)

        parent.to("cuda")
        self.assertEqual(model._encoder_parameter_device(cpu_batch).type, "cuda")
        _, _, logits, _ = parent.cell(cpu_batch)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(model.encoder.direct_to_calls, [])
        self.assertEqual(cpu_batch.device.type, "cpu")
        self.assertEqual(logits.device.type, "cuda")

        parent.to("cpu")
        cuda_batch = torch.randn(2, _INPUT_DIM, device="cuda")
        recon, _, cpu_logits, _ = parent.cell(cuda_batch)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(model._encoder_parameter_device(cuda_batch), _CPU)
        self.assertEqual(cuda_batch.device.type, "cuda")
        self.assertEqual(recon.device.type, "cpu")
        torch.testing.assert_close(cpu_logits, _cpu_encoder_logits(model, cuda_batch))

        parent.to("cuda")
        _, _, again, _ = parent.cell(cpu_batch)
        self.assertEqual(model.direct_to_calls, [])
        self.assertEqual(model._encoder_parameter_device(cpu_batch).type, "cuda")
        self.assertEqual(again.device.type, "cuda")
        self.assertEqual(cpu_batch.device.type, "cpu")

    def test_encoder_only_cuda_matches_cpu_logits(self):
        """Encoder-only CUDA placement must align ``_encode`` and match CPU weights."""
        encoder = RecordingEncoder(_INPUT_DIM, _HIDDEN_DIM, _LATENT_DIM, "layer")
        model = _make_model(encoder=encoder)
        _move_module_to(model.encoder, torch.device("cuda"))
        batch = torch.randn(2, _INPUT_DIM)

        logits, _, _ = model._encode(batch)

        self.assertEqual(encoder.direct_to_calls, [])
        self.assertEqual(batch.device.type, "cpu")
        self.assertEqual(next(model.encoder.parameters()).device.type, "cuda")
        self.assertEqual(next(model.decoder.parameters()).device.type, "cpu")
        self.assertEqual(encoder.seen_batches[-1].device.type, "cuda")
        self.assertEqual(logits.device.type, "cuda")
        torch.testing.assert_close(
            logits.detach().cpu(), _cpu_encoder_logits(model, batch)
        )


if __name__ == "__main__":
    unittest.main()
