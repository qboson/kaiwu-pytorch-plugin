"""Regression tests for CellQVAE decoder-parameter device alignment.

``CellQVAE`` overrides ``QVAE.forward`` and does not call the base method, so
library ``QVAE.forward`` (#1370) does not cover this reconstruction. A parent
``nn.Module.to`` moves the decoder through ``_apply`` and does not call
``QVAEDecoder.to`` (merged #164). Encoder latents and the Bernoulli
``_train_bias`` buffer stay behind. ``forward`` rereads the live decoder
parameter device on every call and copies those tensors onto it.

This file follows ``tests/test_bm_device_sync.py``: meta-device mechanism
tests on every machine, plus CUDA coverage behind ``skipUnless``. The encoder
stays on the caller device so a failure is inside ``forward``, not ``_encode``.

Out of scope: ``_encode`` / ``_encoder_parameter_device`` (open #1371, same
file, different methods), library BM paths (#1348), and the qvae_cell MPS
selector (#512). ``.to()`` cannot copy data out of ``meta``; return trips from
``meta`` retarget storage with ``_apply``.
"""

# Parent moves go through ``_apply`` (see merged #164).
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
    spec = importlib.util.spec_from_file_location("qvae_cell_models_forward", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qvae_cell_models_forward"] = module
    spec.loader.exec_module(module)
    return module


_CELL = _load_cell_models()
CellQVAE = _CELL.CellQVAE
QVAEDecoder = _CELL.QVAEDecoder
QVAEEncoder = _CELL.QVAEEncoder

_INPUT_DIM = 4
_HIDDEN_DIM = 6
_LATENT_DIM = 4
_CPU = torch.device("cpu")
_META = torch.device("meta")


def _config(loss_type):
    """Build the small config ``CellQVAE.forward`` reads."""
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
    decoder=None,
):
    """Return a tiny ``CellQVAE`` whose encoder can stay on the caller device."""
    if encoder is None:
        encoder = QVAEEncoder(_INPUT_DIM, _HIDDEN_DIM, _LATENT_DIM, "layer")
    if decoder is None:
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


class RecordingDecoder(QVAEDecoder):
    """Decoder that records inputs and direct ``.to()`` calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_inputs = []
        self.direct_to_calls = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)

    def forward(self, zeta):
        """Record ``zeta``, then decode."""
        self.seen_inputs.append(zeta.detach())
        return super().forward(zeta)


class ParameterFreeDecoder(nn.Module):  # pylint: disable=too-few-public-methods
    """Decoder stand-in with no parameters."""

    def forward(self, zeta):  # pylint: disable=missing-function-docstring
        return zeta


class ParameterFreeEncoder(nn.Module):  # pylint: disable=too-few-public-methods
    """Encoder stand-in whose output shape matches the latent width."""

    def forward(self, inputs):  # pylint: disable=missing-function-docstring
        return inputs


def _recording_decoder(n_batches=0):
    """Return a decoder that records ``.to()`` and its inputs."""
    return RecordingDecoder(
        _LATENT_DIM + n_batches,
        _HIDDEN_DIM,
        _INPUT_DIM,
        "layer",
    )


def _move_module_to(module, device):
    """Move parameters the way a parent does: ``_apply``, not ``.to``."""
    module._apply(lambda tensor: tensor.to(device))


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


def _move_train_bias(model, device):
    """Move only ``_train_bias``, leaving parameters where they are."""
    model._buffers["_train_bias"] = model._train_bias.detach().to(device)


def _assert_unchanged(case, tensor, before, pointer):
    """Assert ``tensor`` still has the same storage, device, and values."""
    case.assertEqual(tensor.data_ptr(), pointer)
    case.assertEqual(tensor.device, before.device)
    torch.testing.assert_close(tensor, before)


def _decoder_recon_on_cpu(model, zeta, batch_idx=None):
    """Replay decoder weights on CPU for the latent ``forward`` returned.

    ``meta`` weights have no data, so callers must use a real device.
    """
    state = {
        key: value.detach().to(device=_CPU).clone()
        for key, value in model.decoder.state_dict().items()
    }
    decoder = QVAEDecoder(
        model.decoder.fc1.in_features,
        model.decoder.fc1.out_features,
        model.decoder.fc2.out_features,
        "layer",
    )
    weight = next(iter(state.values()))
    decoder.to(dtype=weight.dtype)
    decoder.load_state_dict(state)
    decoder.eval()
    latent = zeta.detach().to(dtype=weight.dtype, device=_CPU)
    if model.n_batches > 0:
        one_hot = torch.nn.functional.one_hot(
            batch_idx.detach().to(device=_CPU),
            num_classes=model.n_batches,
        ).to(dtype=weight.dtype)
        latent = torch.cat([latent, one_hot], dim=-1)
    recon = decoder(latent)
    if model.config.loss_type == "bernoulli":
        recon = recon + model._train_bias.detach().to(dtype=recon.dtype, device=_CPU)
    return recon


def _reject_qvae_forward(*args, **kwargs):
    """Fail when ``CellQVAE.forward`` delegates to ``QVAE.forward``."""
    del args, kwargs
    raise AssertionError("CellQVAE.forward called QVAE.forward")


class TestCellQVAEForwardParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_parent_to_does_not_call_decoder_to(self):
        """``nn.Module.to`` must move a decoder without calling its ``.to``."""
        decoder = _recording_decoder()
        parent = nn.Module()
        parent.decoder = decoder

        parent.to(_META)

        self.assertEqual(decoder.direct_to_calls, [])
        self.assertEqual(next(decoder.parameters()).device, _META)

    def test_decoder_only_apply_accepts_cpu_batch(self):
        """CPU latents must reconstruct on the decoder parameter device."""
        decoder = _recording_decoder()
        model = _make_model(decoder=decoder)
        _move_module_to(model.decoder, _META)
        batch = torch.ones(2, _INPUT_DIM)
        before = batch.clone()
        pointer = batch.data_ptr()

        recon, _, logits, zeta = model(batch)

        self.assertEqual(decoder.direct_to_calls, [])
        _assert_unchanged(self, batch, before, pointer)
        self.assertEqual(next(model.encoder.parameters()).device, _CPU)
        self.assertEqual(next(model.decoder.parameters()).device, _META)
        self.assertEqual(model._decoder_parameter_device(zeta), _META)
        self.assertEqual(logits.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.device, _META)
        self.assertEqual(recon.dtype, torch.float32)
        self.assertEqual(recon.shape, (2, _INPUT_DIM))

    def test_bernoulli_bias_follows_decoder_and_stays_put(self):
        """A CPU ``_train_bias`` is added on the decoder device, not mutated."""
        model = _make_model("bernoulli")
        model.set_train_bias(0.25)
        bias_before = model._train_bias.clone()
        bias_pointer = model._train_bias.data_ptr()
        _move_module_to(model.decoder, _META)
        batch = torch.zeros(2, _INPUT_DIM)
        before = batch.clone()
        pointer = batch.data_ptr()

        recon, _, _, zeta = model(batch)

        _assert_unchanged(self, batch, before, pointer)
        _assert_unchanged(self, model._train_bias, bias_before, bias_pointer)
        self.assertEqual(model._train_bias.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.device, _META)
        self.assertEqual(recon.shape, (2, _INPUT_DIM))

    def test_mse_does_not_add_train_bias(self):
        """MSE reconstruction must ignore a ``_train_bias`` left on meta."""
        model = _make_model("mse")
        _move_train_bias(model, _META)
        batch = torch.ones(2, _INPUT_DIM)

        recon, _, _, _ = model(batch)

        self.assertEqual(model._train_bias.device, _META)
        self.assertEqual(recon.device, _CPU)
        self.assertEqual(recon.shape, (2, _INPUT_DIM))

    def test_batch_one_hot_follows_decoder_parameters(self):
        """Batch indices stay put; the one-hot must join the decoder device."""
        decoder = _recording_decoder(n_batches=3)
        model = _make_model(n_batches=3, decoder=decoder)
        _move_module_to(model.decoder, _META)
        batch = torch.ones(2, _INPUT_DIM)
        batch_idx = torch.tensor([0, 2])
        before = batch_idx.clone()
        pointer = batch_idx.data_ptr()

        recon, _, _, zeta = model(batch, batch_idx)

        self.assertEqual(decoder.direct_to_calls, [])
        _assert_unchanged(self, batch_idx, before, pointer)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(decoder.seen_inputs[-1].device, _META)
        self.assertEqual(decoder.seen_inputs[-1].shape, (2, _LATENT_DIM + 3))
        self.assertEqual(recon.device, _META)
        self.assertEqual(recon.shape, (2, _INPUT_DIM))

    def test_float64_batch_keeps_dtype_on_decoder_device(self):
        """Device alignment must not downcast a float64 batch or its one-hot."""
        model = _make_model(dtype=torch.float64)
        _move_module_to(model.decoder, _META)
        batch = torch.ones(2, _INPUT_DIM, dtype=torch.float64)

        recon, _, logits, zeta = model(batch)

        self.assertEqual(next(model.decoder.parameters()).dtype, torch.float64)
        self.assertEqual(recon.dtype, torch.float64)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(zeta.dtype, torch.float64)
        self.assertEqual(recon.device, _META)

        conditional = _make_model(dtype=torch.float64, n_batches=3)
        _move_module_to(conditional.decoder, _META)
        batch_idx = torch.tensor([0, 2])
        recon, _, _, zeta = conditional(batch, batch_idx)

        self.assertEqual(batch_idx.device, _CPU)
        self.assertEqual(zeta.dtype, torch.float64)
        self.assertEqual(recon.dtype, torch.float64)
        self.assertEqual(recon.device, _META)
        self.assertEqual(recon.shape, (2, _INPUT_DIM))

    def test_dtype_only_move_keeps_cpu_device(self):
        """``.to(dtype=...)`` must keep CPU and must not change the reconstruction."""
        model = _make_model("bernoulli")
        model.set_train_bias(0.25)
        model.to(dtype=torch.float64)
        batch = torch.ones(2, _INPUT_DIM, dtype=torch.float64)

        recon, _, logits, zeta = model(batch)

        self.assertEqual(next(model.decoder.parameters()).dtype, torch.float64)
        self.assertEqual(next(model.decoder.parameters()).device, _CPU)
        self.assertEqual(model._train_bias.dtype, torch.float64)
        self.assertEqual(model._train_bias.device, _CPU)
        self.assertEqual(recon.dtype, torch.float64)
        self.assertEqual(recon.device, _CPU)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(logits.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        torch.testing.assert_close(recon, _decoder_recon_on_cpu(model, zeta))

    def test_parameter_device_is_reread_after_later_moves(self):
        """The helper must be reread after each decoder move, not cached."""
        model = _make_model()
        batch = torch.randn(2, _INPUT_DIM)
        pointer = batch.data_ptr()

        self.assertEqual(model._decoder_parameter_device(batch), _CPU)
        recon, _, logits, zeta = model(batch)
        self.assertEqual(recon.device, _CPU)
        self.assertEqual(logits.device, _CPU)
        self.assertEqual(zeta.device, _CPU)

        _move_module_to(model.decoder, _META)
        self.assertEqual(model._decoder_parameter_device(batch), _META)
        recon, _, logits, zeta = model(batch)
        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.device, _CPU)
        self.assertEqual(next(model.encoder.parameters()).device, _CPU)
        self.assertEqual(logits.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.device, _META)

        # ``.to(cpu)`` cannot copy out of meta; ``_apply`` still retargets.
        model.decoder._apply(_retarget_storage(_CPU))
        self.assertEqual(model._decoder_parameter_device(batch), _CPU)
        recon, _, _, zeta = model(batch)
        self.assertEqual(next(model.encoder.parameters()).device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.device, _CPU)
        self.assertEqual(batch.device, _CPU)

    def test_encoder_parameters_do_not_choose_the_decoder_device(self):
        """A meta encoder must not decide where ``forward`` reconstructs.

        ``forward`` is not called: a meta encoder rejects a CPU batch inside
        ``_encode``, which is open #1371. This test only locks the helper.
        """
        model = _make_model()
        fallback = torch.zeros(1, device=_CPU)
        _move_module_to(model.encoder, _META)

        self.assertEqual(next(model.encoder.parameters()).device, _META)
        self.assertEqual(next(model.decoder.parameters()).device, _CPU)
        self.assertEqual(model._decoder_parameter_device(fallback), _CPU)

    def test_parameter_free_decoder_uses_the_latent_device(self):
        """With no decoder parameters, reconstruction stays on the latent device."""
        model = _make_model(
            encoder=ParameterFreeEncoder(),
            decoder=ParameterFreeDecoder(),
        )
        batch = torch.randn(2, _INPUT_DIM)
        before = batch.clone()
        pointer = batch.data_ptr()

        self.assertEqual(list(model.decoder.parameters()), [])
        self.assertEqual(model._decoder_parameter_device(batch), _CPU)
        recon, _, logits, zeta = model(batch)

        _assert_unchanged(self, batch, before, pointer)
        self.assertEqual(logits.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.device, _CPU)
        torch.testing.assert_close(recon, zeta)

        meta_batch = torch.ones(2, _INPUT_DIM, device=_META)
        self.assertEqual(model._decoder_parameter_device(meta_batch), _META)
        meta_recon, _, meta_logits, meta_zeta = model(meta_batch)
        self.assertEqual(meta_batch.device, _META)
        self.assertEqual(meta_logits.device, _META)
        self.assertEqual(meta_zeta.device, _META)
        self.assertEqual(meta_recon.device, _META)
        self.assertEqual(meta_recon.shape, meta_batch.shape)

    def test_same_device_recon_matches_the_decoder(self):
        """A CPU batch already on the decoder device must be unchanged."""
        model = _make_model()
        batch = torch.randn(2, _INPUT_DIM)
        before = batch.clone()
        pointer = batch.data_ptr()

        recon, _, _, zeta = model(batch)

        _assert_unchanged(self, batch, before, pointer)
        self.assertEqual(recon.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        torch.testing.assert_close(recon, _decoder_recon_on_cpu(model, zeta))

    def test_same_device_bernoulli_batch_matches_the_decoder(self):
        """Bernoulli bias and the batch one-hot must match a CPU decoder."""
        model = _make_model("bernoulli", n_batches=3)
        model.set_train_bias(0.25)
        batch = torch.rand(2, _INPUT_DIM)
        batch_idx = torch.tensor([1, 0])
        before = batch.clone()
        pointer = batch.data_ptr()
        index_before = batch_idx.clone()
        index_pointer = batch_idx.data_ptr()
        bias_before = model._train_bias.clone()
        bias_pointer = model._train_bias.data_ptr()

        recon, _, _, zeta = model(batch, batch_idx)

        _assert_unchanged(self, batch, before, pointer)
        _assert_unchanged(self, batch_idx, index_before, index_pointer)
        _assert_unchanged(self, model._train_bias, bias_before, bias_pointer)
        self.assertEqual(recon.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        torch.testing.assert_close(
            recon,
            _decoder_recon_on_cpu(model, zeta, batch_idx),
        )

    def test_same_device_alignment_keeps_autograd(self):
        """``.to`` on an already-aligned latent must not detach the graph."""
        model = _make_model()
        model.train()
        model.encoder.dropout.p = 0
        model.decoder.dropout.p = 0
        batch = torch.randn(2, _INPUT_DIM, requires_grad=True)

        recon, _, _, zeta = model(batch)
        recon.sum().backward()

        weight = model.decoder.fc1.weight
        self.assertEqual(zeta.device, _CPU)
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
        _move_module_to(model.decoder, _META)
        batch = torch.ones(2, _INPUT_DIM)

        with unittest.mock.patch.object(_CELL.QVAE, "forward", _reject_qvae_forward):
            recon, _, logits, zeta = model(batch)

        self.assertEqual(recon.device, _META)
        self.assertEqual(logits.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.shape, (2, _INPUT_DIM))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCellQVAEForwardParameterDeviceCuda(unittest.TestCase):
    """Decoder-only CUDA coverage. The encoder stays on CPU."""

    def test_parent_apply_skips_decoder_to_and_matches_cpu_decoder(self):
        """``_apply(cuda)`` must not call ``.to`` and must match CPU weights."""
        decoder = _recording_decoder()
        model = _make_model("bernoulli", decoder=decoder)
        model.set_train_bias(0.25)
        _move_module_to(model.decoder, torch.device("cuda"))
        batch = torch.rand(2, _INPUT_DIM)
        before = batch.clone()
        pointer = batch.data_ptr()
        bias_before = model._train_bias.clone()
        bias_pointer = model._train_bias.data_ptr()

        recon, _, logits, zeta = model(batch)

        self.assertEqual(decoder.direct_to_calls, [])
        _assert_unchanged(self, batch, before, pointer)
        _assert_unchanged(self, model._train_bias, bias_before, bias_pointer)
        self.assertEqual(next(model.encoder.parameters()).device, _CPU)
        self.assertEqual(logits.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(model._train_bias.device, _CPU)
        self.assertEqual(recon.device.type, "cuda")
        torch.testing.assert_close(
            recon.detach().cpu(),
            _decoder_recon_on_cpu(model, zeta),
        )

    def test_backward_from_cpu_batch_reaches_cuda_decoder(self):
        """The latent copy inside ``forward`` must stay in the autograd graph."""
        model = _make_model()
        model.train()
        model.encoder.dropout.p = 0
        model.decoder.dropout.p = 0
        _move_module_to(model.decoder, torch.device("cuda"))
        batch = torch.randn(2, _INPUT_DIM, requires_grad=True)

        recon, _, _, zeta = model(batch)
        recon.sum().backward()

        weight = model.decoder.fc1.weight
        self.assertEqual(recon.device.type, "cuda")
        self.assertEqual(zeta.device.type, "cpu")
        self.assertIsNotNone(batch.grad)
        self.assertEqual(batch.grad.device.type, "cpu")
        self.assertEqual(batch.grad.shape, batch.shape)
        self.assertTrue(torch.isfinite(batch.grad).all().item())
        self.assertIsNotNone(weight.grad)
        self.assertEqual(weight.grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(weight.grad).all().item())
        self.assertGreater(float(weight.grad.abs().sum()), 0.0)

    def test_decoder_device_is_reread_across_cuda_and_cpu(self):
        """CUDA and CPU decoder moves must be reread, not cached."""
        decoder = _recording_decoder()
        model = _make_model(decoder=decoder)
        batch = torch.randn(2, _INPUT_DIM)
        cuda = torch.device("cuda")

        _move_module_to(model.decoder, cuda)
        self.assertEqual(model._decoder_parameter_device(batch).type, "cuda")
        recon, _, _, zeta = model(batch)
        self.assertEqual(batch.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.device.type, "cuda")
        torch.testing.assert_close(
            recon.detach().cpu(),
            _decoder_recon_on_cpu(model, zeta),
        )

        _move_module_to(model.decoder, _CPU)
        self.assertEqual(model._decoder_parameter_device(batch), _CPU)
        recon, _, _, zeta = model(batch)
        self.assertEqual(recon.device, _CPU)
        self.assertEqual(zeta.device, _CPU)
        torch.testing.assert_close(recon, _decoder_recon_on_cpu(model, zeta))

        _move_module_to(model.decoder, cuda)
        self.assertEqual(decoder.direct_to_calls, [])
        self.assertEqual(model._decoder_parameter_device(batch).type, "cuda")
        self.assertEqual(next(model.encoder.parameters()).device, _CPU)
        recon, _, _, zeta = model(batch)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.device.type, "cuda")

    def test_cuda_latent_follows_cpu_decoder(self):
        """A CUDA latent must still reconstruct on a CPU decoder."""
        model = _make_model(encoder=ParameterFreeEncoder())
        batch = torch.ones(2, _INPUT_DIM, device="cuda")
        pointer = batch.data_ptr()

        recon, _, logits, zeta = model(batch)

        self.assertEqual(batch.data_ptr(), pointer)
        self.assertEqual(batch.device.type, "cuda")
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(zeta.device.type, "cuda")
        self.assertEqual(next(model.decoder.parameters()).device, _CPU)
        self.assertEqual(recon.device, _CPU)
        torch.testing.assert_close(recon, _decoder_recon_on_cpu(model, zeta))

    def test_cuda_bias_follows_cpu_decoder(self):
        """A CUDA ``_train_bias`` must be added on the CPU decoder device."""
        model = _make_model("bernoulli")
        model.set_train_bias(0.25)
        _move_train_bias(model, torch.device("cuda"))
        bias_before = model._train_bias.detach().cpu().clone()
        bias_pointer = model._train_bias.data_ptr()
        batch = torch.rand(2, _INPUT_DIM)

        recon, _, _, zeta = model(batch)

        self.assertEqual(model._train_bias.device.type, "cuda")
        self.assertEqual(model._train_bias.data_ptr(), bias_pointer)
        torch.testing.assert_close(model._train_bias.detach().cpu(), bias_before)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.device, _CPU)
        torch.testing.assert_close(recon, _decoder_recon_on_cpu(model, zeta))

    def test_batch_one_hot_and_bias_match_on_cuda(self):
        """One-hot columns follow the CUDA decoder; ids and bias stay on CPU."""
        decoder = _recording_decoder(n_batches=3)
        model = _make_model("bernoulli", n_batches=3, decoder=decoder)
        model.set_train_bias(0.25)
        _move_module_to(model.decoder, torch.device("cuda"))
        batch = torch.rand(2, _INPUT_DIM)
        batch_idx = torch.tensor([0, 2])
        index_pointer = batch_idx.data_ptr()
        bias_pointer = model._train_bias.data_ptr()

        recon, _, _, zeta = model(batch, batch_idx)

        seen = decoder.seen_inputs[-1]
        self.assertEqual(decoder.direct_to_calls, [])
        self.assertEqual(seen.device.type, "cuda")
        self.assertEqual(seen.shape, (2, _LATENT_DIM + 3))
        torch.testing.assert_close(
            seen[:, _LATENT_DIM:].detach().cpu(),
            torch.nn.functional.one_hot(batch_idx, num_classes=3).float(),
        )
        self.assertEqual(batch_idx.device, _CPU)
        self.assertEqual(batch_idx.data_ptr(), index_pointer)
        self.assertEqual(model._train_bias.device, _CPU)
        self.assertEqual(model._train_bias.data_ptr(), bias_pointer)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.device.type, "cuda")
        torch.testing.assert_close(
            recon.detach().cpu(),
            _decoder_recon_on_cpu(model, zeta, batch_idx),
        )

    def test_float64_cuda_decoder_keeps_dtype(self):
        """A float64 decoder move must not downcast the reconstruction."""
        model = _make_model(dtype=torch.float64)
        _move_module_to(model.decoder, torch.device("cuda"))
        batch = torch.ones(2, _INPUT_DIM, dtype=torch.float64)

        recon, _, logits, zeta = model(batch)

        self.assertEqual(recon.dtype, torch.float64)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(zeta.dtype, torch.float64)
        self.assertEqual(zeta.device, _CPU)
        self.assertEqual(recon.device.type, "cuda")
        torch.testing.assert_close(
            recon.detach().cpu(),
            _decoder_recon_on_cpu(model, zeta),
        )


if __name__ == "__main__":
    unittest.main()
