"""Regression tests for QVAE reconstruction device alignment.

``QVAE.forward`` runs the encoder, the decoder, and the Bernoulli
``_train_bias`` buffer as three different pieces of module state. A parent
``_apply`` (the path ``nn.Module.to`` uses for children; merged #164) moves
whichever of those pieces actually own parameters or buffers, while a caller
batch often stays on CPU. Reconstruction follows the live parameter device
of the submodule it is about to touch. ``_parameter_device`` is reread on
every call and is not a cache. A parameter-free submodule falls back to the
activation already in hand.

The stale location for this path is the Bernoulli buffer: moving only the
decoder leaves ``_train_bias`` behind, and that buffer device must not choose
the reconstruction device.

CPU cases use ``meta``, the same stand-in as ``tests/test_bm_device_sync.py``.
CUDA cases are skipped unless a real accelerator is present.

Open #1348 aligns BM-facing latents in ``energy`` / ``bm_loss`` /
``_cross_entropy``. This file does not cover those paths. KL ``entropy.to``
(#1268) and ``_weight_decay_loss`` stay on the BM, so device-split ``loss``
tests replace those two terms and check the reconstruction term only.
"""

# Parent moves go through ``_apply`` (see #164).
# pylint: disable=protected-access

import unittest
from contextlib import contextmanager
from unittest import mock

import torch
from torch import nn

from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae import MixtureGeneric, QVAE
from kaiwu.torch_plugin.qvae_dist_util import FactorialBernoulliUtil

INPUT_DIM = 8
LATENT_DIM = 4
META = torch.device("meta")
CPU = torch.device("cpu")


class RecordsModuleMoves(nn.Module):
    """Remember direct ``.to()`` calls and the device of each input."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_devices: list[torch.device] = []
        self.seen_batches: list[torch.Tensor] = []
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map one batch. Subclasses replace this."""
        del inputs
        raise NotImplementedError

    def _remember(self, inputs: torch.Tensor) -> torch.Tensor:
        """Store ``inputs`` device and return ``inputs`` unchanged."""
        self.seen_devices.append(inputs.device)
        self.seen_batches.append(inputs.detach())
        return inputs


class LinearEncoder(RecordsModuleMoves):
    """Encoder with one linear map, so the parameter device is observable."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Record the input device and return latent logits."""
        self._remember(inputs)
        return self.fc(inputs)


class LinearDecoder(RecordsModuleMoves):
    """Decoder with one linear map, so the parameter device is observable."""

    def __init__(self, latent_dim: int, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Record the latent device and return reconstruction logits."""
        self._remember(inputs)
        return self.fc(inputs)


class ParameterFreeMap(RecordsModuleMoves):
    """Batch map with no parameters, covering the device fallback."""

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.out_features = out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Record the input device and return zeros on that device."""
        self._remember(inputs)
        return torch.zeros(
            inputs.shape[0],
            self.out_features,
            device=inputs.device,
            dtype=inputs.dtype,
        )


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


def _build_qvae(
    loss_type: str = "mse",
    encoder: nn.Module | None = None,
    decoder: nn.Module | None = None,
) -> RecordingQVAE:
    """Return a small QVAE whose encoder and decoder own real parameters.

    ``encoder`` / ``decoder`` override the linear maps when a test needs a
    parameter-free submodule.
    """
    config = type(
        "Config",
        (),
        {
            "num_latent_units": LATENT_DIM,
            "loss_type": loss_type,
            "dist_beta": 1.0,
            "kl_beta": 0.0,
            "weight_decay": 0.0,
        },
    )()
    return RecordingQVAE(
        input_dimension=INPUT_DIM,
        activation_fct=None,
        config=config,
        encoder=(
            encoder if encoder is not None else LinearEncoder(INPUT_DIM, LATENT_DIM)
        ),
        decoder=(
            decoder if decoder is not None else LinearDecoder(LATENT_DIM, INPUT_DIM)
        ),
        bm=RestrictedBoltzmannMachine(LATENT_DIM // 2, LATENT_DIM // 2),
        sampler=object(),
    )


def _to_meta(tensor: torch.Tensor) -> torch.Tensor:
    """``_apply`` callback that moves storage onto ``meta``."""
    return tensor.to(META)


def _retarget_storage(device: torch.device):
    """Return an ``_apply`` function that places empty storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    """

    def _apply_fn(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == device:
            return tensor
        return torch.empty(tensor.shape, dtype=tensor.dtype, device=device)

    return _apply_fn


def _sigmoid_zeta(posterior, is_training: bool) -> torch.Tensor:
    """Replace Bernoulli reparameterization with a deterministic map."""
    del is_training
    return torch.sigmoid(posterior.logit_mu)


@contextmanager
def _deterministic_posterior():
    """Make reconstruction comparable across devices."""
    with mock.patch.object(MixtureGeneric, "reparameterize", _sigmoid_zeta):
        yield


_MSE_LOSS = torch.nn.functional.mse_loss


def _spy_mse(seen: dict):
    """Record the devices that reach ``mse_loss`` and call the real function."""

    def _wrapped(pred, target, reduction="mean"):
        seen["pred"] = pred.device
        seen["target"] = target.device
        return _MSE_LOSS(pred, target, reduction=reduction)

    return _wrapped


def _spy_bernoulli(seen: dict):
    """Record the devices that reach Bernoulli log-prob and call the real method."""
    real = FactorialBernoulliUtil.log_prob_per_var

    def _wrapped(dist, samples):
        seen["logits"] = dist.logit_mu.device
        seen["samples"] = samples.device
        return real(dist, samples)

    return _wrapped


@contextmanager
def _patch_aux_losses(qvae: QVAE, device: torch.device, batch_size: int):
    """Keep KL and weight decay off the reconstruction device split.

    Those terms follow the BM (#1348 / #1268), not the decoder. Zeros are
    already on the decoder device so the sum can be checked, and so the
    test does not call ``sample()``. Callers that care about the
    reconstruction term spy on ``mse_loss`` or Bernoulli log-prob: adding a
    meta zero would otherwise hide a CPU reconstruction term.
    """
    kl_term = torch.zeros(batch_size, device=device)
    weight_decay = torch.zeros((), device=device)
    with mock.patch.object(qvae, "_kl_dist_from", return_value=kl_term):
        with mock.patch.object(qvae, "_weight_decay_loss", return_value=weight_decay):
            yield


def _cpu_reference_recon(qvae: QVAE, inputs: torch.Tensor) -> torch.Tensor:
    """Reconstruction of ``inputs`` against a CPU copy of the same weights."""
    reference = _build_qvae(qvae.config.loss_type)
    state = {
        key: value.detach().to(device=CPU).clone()
        for key, value in qvae.state_dict().items()
    }
    reference.load_state_dict(state)
    reference.eval()
    with _deterministic_posterior():
        recon, _, _, _ = reference(inputs.detach().to(device=CPU))
    return recon


def _assert_child_to_not_called(test: unittest.TestCase, qvae: RecordingQVAE):
    """Parent moves and forward must not call submodule ``.to()``."""
    test.assertEqual(qvae.direct_to_calls, [])
    test.assertEqual(qvae.encoder.direct_to_calls, [])
    test.assertEqual(qvae.decoder.direct_to_calls, [])


class TestQvaeReconstructionParameterDevice(unittest.TestCase):
    """Meta-device tests that run on CPU-only machines."""

    def test_parent_apply_move_accepts_cpu_batch(self):
        """CPU batches must reconstruct on the parameter device after ``_apply``."""
        parent = nn.Module()
        parent.qvae = _build_qvae("mse")
        parent.qvae.eval()
        inputs = torch.randn(2, INPUT_DIM)
        parent._apply(_to_meta)

        recon_x, _, logits, _ = parent.qvae(inputs)

        _assert_child_to_not_called(self, parent.qvae)
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(next(parent.qvae.encoder.parameters()).device, META)
        self.assertEqual(next(parent.qvae.decoder.parameters()).device, META)
        self.assertEqual(parent.qvae.encoder.seen_devices[-1], META)
        self.assertEqual(parent.qvae.decoder.seen_devices[-1], META)
        self.assertEqual(logits.device, META)
        self.assertEqual(recon_x.device, META)
        self.assertEqual(recon_x.shape, (2, INPUT_DIM))

    def test_decoder_to_meta_aligns_latents_without_another_move(self):
        """A direct ``decoder.to(meta)`` moves latents, not the module, again."""
        qvae = _build_qvae("mse")
        qvae.eval()
        qvae.decoder.to(META)
        inputs = torch.randn(2, INPUT_DIM)
        calls_after_setup = list(qvae.decoder.direct_to_calls)

        recon_x, _, logits, _ = qvae(inputs)

        self.assertEqual(qvae.decoder.direct_to_calls, calls_after_setup)
        self.assertEqual(qvae.encoder.direct_to_calls, [])
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(next(qvae.encoder.parameters()).device, CPU)
        self.assertEqual(next(qvae.decoder.parameters()).device, META)
        self.assertEqual(qvae.encoder.seen_devices[-1], CPU)
        self.assertEqual(qvae.decoder.seen_devices[-1], META)
        self.assertEqual(logits.device, CPU)
        self.assertEqual(recon_x.device, META)
        self.assertEqual(recon_x.shape, inputs.shape)

    def test_decoder_only_move_aligns_latents_and_train_bias(self):
        """Moving only the decoder must not add a CPU bias to meta logits.

        ``_apply`` leaves ``_train_bias`` on CPU. That stale buffer device
        must not become the reconstruction device.
        """
        qvae = _build_qvae("bernoulli")
        qvae.eval()
        qvae.set_train_bias(0.25)
        inputs = torch.zeros(2, INPUT_DIM)
        qvae.decoder._apply(_to_meta)

        recon_x, _, logits, _ = qvae(inputs)

        _assert_child_to_not_called(self, qvae)
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(qvae._train_bias.device, CPU)
        self.assertEqual(logits.device, CPU)
        self.assertEqual(qvae.decoder.seen_devices[-1], META)
        self.assertEqual(recon_x.device, META)
        self.assertEqual(recon_x.shape, inputs.shape)
        self.assertEqual(next(qvae.decoder.parameters()).device, META)

    def test_loss_aligns_cpu_targets_with_decoder_parameters(self):
        """``loss`` must score CPU targets and logits on the decoder device."""
        qvae = _build_qvae("mse")
        qvae.eval()
        qvae.decoder._apply(_to_meta)
        targets = torch.ones(2, INPUT_DIM)
        recon_x = torch.zeros(2, INPUT_DIM)
        targets_before = targets.clone()
        recon_before = recon_x.clone()
        seen = {}

        with _patch_aux_losses(qvae, META, targets.shape[0]):
            with mock.patch("kaiwu.torch_plugin.qvae.F.mse_loss", _spy_mse(seen)):
                loss = qvae.loss(targets, recon_x, posterior=None)

        self.assertEqual(seen["pred"], META)
        self.assertEqual(seen["target"], META)
        self.assertEqual(loss.device, META)
        self.assertEqual(loss.shape, ())
        self.assertEqual(targets.device, CPU)
        self.assertEqual(recon_x.device, CPU)
        torch.testing.assert_close(targets, targets_before)
        torch.testing.assert_close(recon_x, recon_before)

    def test_bernoulli_loss_aligns_cpu_targets(self):
        """Bernoulli reconstruction loss must land on the decoder device."""
        qvae = _build_qvae("bernoulli")
        qvae.eval()
        qvae.decoder._apply(_to_meta)
        targets = torch.rand(2, INPUT_DIM)
        recon_x = torch.randn(2, INPUT_DIM)
        seen = {}

        with _patch_aux_losses(qvae, META, targets.shape[0]):
            with mock.patch.object(
                FactorialBernoulliUtil,
                "log_prob_per_var",
                _spy_bernoulli(seen),
            ):
                loss = qvae.loss(targets, recon_x, posterior=None)

        self.assertEqual(seen["logits"], META)
        self.assertEqual(seen["samples"], META)
        self.assertEqual(loss.device, META)
        self.assertEqual(loss.shape, ())
        self.assertEqual(targets.device, CPU)
        self.assertEqual(recon_x.device, CPU)

    def test_mse_loss_matches_formula_when_already_aligned(self):
        """Device alignment must not change the MSE reduction on one device."""
        qvae = _build_qvae("mse")
        targets = torch.rand(2, INPUT_DIM)
        recon_x = torch.randn(2, INPUT_DIM)

        with _patch_aux_losses(qvae, CPU, targets.shape[0]):
            loss = qvae.loss(targets, recon_x, posterior=None)

        expected = torch.nn.functional.mse_loss(
            recon_x, targets, reduction="sum"
        ) / targets.size(0)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(loss.device, CPU)

    def test_bernoulli_loss_matches_log_prob_when_already_aligned(self):
        """Device alignment must not change the Bernoulli reconstruction term."""
        qvae = _build_qvae("bernoulli")
        targets = torch.rand(2, INPUT_DIM)
        recon_x = torch.randn(2, INPUT_DIM)

        with _patch_aux_losses(qvae, CPU, targets.shape[0]):
            loss = qvae.loss(targets, recon_x, posterior=None)

        expected = (
            -FactorialBernoulliUtil(recon_x).log_prob_per_var(targets).sum(dim=1).mean()
        )
        torch.testing.assert_close(loss, expected)
        self.assertEqual(loss.device, CPU)

    def test_float64_batch_keeps_dtype_after_apply_move(self):
        """A float64 batch must not be downcast when parameters move to meta."""
        qvae = _build_qvae("mse")
        qvae.double()
        qvae.eval()
        qvae._apply(_to_meta)

        recon_x, _, logits, _ = qvae(torch.ones(2, INPUT_DIM, dtype=torch.float64))

        self.assertEqual(qvae.encoder.fc.weight.dtype, torch.float64)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(recon_x.dtype, torch.float64)
        self.assertEqual(recon_x.device, META)

    def test_dtype_only_move_keeps_cpu_device(self):
        """``.to(dtype=...)`` must not change the device the helper reports."""
        qvae = _build_qvae("bernoulli")
        qvae.eval()
        qvae.set_train_bias(0.25)
        qvae.to(dtype=torch.float64)
        inputs = torch.ones(2, INPUT_DIM, dtype=torch.float64)

        recon_x, _, logits, _ = qvae(inputs)

        self.assertEqual(qvae.encoder.fc.weight.dtype, torch.float64)
        self.assertEqual(qvae.decoder.fc.weight.dtype, torch.float64)
        self.assertEqual(qvae._train_bias.dtype, torch.float64)
        self.assertEqual(qvae._train_bias.device, CPU)
        self.assertEqual(qvae._parameter_device(qvae.encoder, inputs), CPU)
        self.assertEqual(qvae._parameter_device(qvae.decoder, inputs), CPU)
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(recon_x.dtype, torch.float64)
        self.assertEqual(recon_x.device, CPU)
        self.assertEqual(inputs.device, CPU)

    def test_cpu_forward_still_matches_shapes(self):
        """An already-aligned CPU batch keeps the public reconstruction contract."""
        qvae = _build_qvae("mse")
        qvae.eval()
        inputs = torch.ones(3, INPUT_DIM)

        recon_x, posterior, logits, zeta = qvae(inputs)

        self.assertEqual(recon_x.device, CPU)
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(logits.shape, (3, LATENT_DIM))
        self.assertEqual(zeta.shape, (3, LATENT_DIM))
        self.assertEqual(posterior.logit_mu.shape, logits.shape)
        self.assertEqual(recon_x.shape, inputs.shape)

    def test_bernoulli_recon_matches_cpu_weight_copy(self):
        """Same-device Bernoulli reconstruction matches a CPU weight copy."""
        qvae = _build_qvae("bernoulli")
        qvae.eval()
        qvae.set_train_bias(0.25)
        inputs = torch.rand(2, INPUT_DIM)

        with _deterministic_posterior():
            recon_x, _, _, zeta = qvae(inputs)

        torch.testing.assert_close(recon_x, _cpu_reference_recon(qvae, inputs))
        self.assertEqual(recon_x.device, CPU)
        self.assertEqual(zeta.device, CPU)
        self.assertEqual(inputs.device, CPU)

    def test_parameter_device_is_reread_after_later_moves(self):
        """The helper must be reread after each move, not cached on first use."""
        parent = nn.Module()
        parent.qvae = _build_qvae("mse")
        parent.qvae.eval()
        inputs = torch.randn(2, INPUT_DIM)

        self.assertEqual(
            parent.qvae._parameter_device(parent.qvae.encoder, inputs), CPU
        )
        first, _, _, _ = parent.qvae(inputs)
        self.assertEqual(first.device, CPU)

        parent.to(META)
        self.assertEqual(
            parent.qvae._parameter_device(parent.qvae.encoder, inputs), META
        )
        second, _, _, _ = parent.qvae(inputs)
        _assert_child_to_not_called(self, parent.qvae)
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(second.device, META)

        # ``.to(cpu)`` cannot copy out of meta; ``_apply`` still retargets.
        parent.qvae._apply(_retarget_storage(CPU))
        self.assertEqual(
            parent.qvae._parameter_device(parent.qvae.decoder, inputs), CPU
        )
        third, _, _, _ = parent.qvae(inputs)
        _assert_child_to_not_called(self, parent.qvae)
        self.assertEqual(third.device, CPU)

        with _patch_aux_losses(parent.qvae, CPU, inputs.shape[0]):
            loss = parent.qvae.loss(inputs, third, posterior=None)
        self.assertEqual(loss.device, CPU)
        self.assertEqual(inputs.device, CPU)

    def test_dataset_mean_follows_encoder_parameters(self):
        """A CPU dataset mean is applied on the encoder parameter device."""
        qvae = _build_qvae("bernoulli")
        qvae.eval()
        mean = torch.full((INPUT_DIM,), 0.25)
        qvae.set_dataset_mean(mean)
        inputs = torch.full((2, INPUT_DIM), 0.75)

        qvae(inputs)

        torch.testing.assert_close(qvae.encoder.seen_batches[-1], inputs - mean)
        self.assertEqual(mean.device, CPU)

        qvae.encoder._apply(_to_meta)
        qvae.decoder._apply(_to_meta)
        qvae(inputs)

        self.assertEqual(mean.device, CPU)
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(qvae.encoder.direct_to_calls, [])
        self.assertEqual(qvae.decoder.direct_to_calls, [])
        self.assertEqual(qvae.encoder.seen_devices[-1], META)
        self.assertEqual(qvae.decoder.seen_devices[-1], META)

    def test_parameter_free_encoder_keeps_the_batch_device(self):
        """With no encoder parameters, the batch device is the fallback."""
        qvae = _build_qvae("mse", encoder=ParameterFreeMap(LATENT_DIM))
        qvae.eval()
        qvae.decoder._apply(_to_meta)
        inputs = torch.randn(2, INPUT_DIM)

        recon_x, _, logits, _ = qvae(inputs)

        self.assertEqual(list(qvae.encoder.parameters()), [])
        self.assertEqual(qvae._parameter_device(qvae.encoder, inputs), CPU)
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(qvae.encoder.seen_devices[-1], CPU)
        self.assertEqual(logits.device, CPU)
        self.assertEqual(qvae.decoder.seen_devices[-1], META)
        self.assertEqual(recon_x.device, META)

    def test_parameter_free_decoder_follows_latent_device(self):
        """With no decoder parameters, latents keep the encoder device."""
        qvae = _build_qvae("mse", decoder=ParameterFreeMap(INPUT_DIM))
        qvae.eval()
        qvae.encoder._apply(_to_meta)
        inputs = torch.randn(2, INPUT_DIM)

        recon_x, _, logits, zeta = qvae(inputs)

        self.assertEqual(list(qvae.decoder.parameters()), [])
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(qvae.encoder.seen_devices[-1], META)
        self.assertEqual(logits.device, META)
        self.assertEqual(zeta.device, META)
        self.assertEqual(qvae._parameter_device(qvae.decoder, zeta), META)
        self.assertEqual(qvae.decoder.seen_devices[-1], META)
        self.assertEqual(recon_x.device, META)

    def test_loss_falls_back_when_decoder_has_no_parameters(self):
        """A parameter-free decoder scores on the reconstruction tensor device."""
        qvae = _build_qvae("mse", decoder=ParameterFreeMap(INPUT_DIM))
        targets = torch.ones(2, INPUT_DIM)
        recon_x = torch.zeros(2, INPUT_DIM, device=META)
        seen = {}

        self.assertEqual(qvae._parameter_device(qvae.decoder, recon_x), META)
        with _patch_aux_losses(qvae, META, targets.shape[0]):
            with mock.patch("kaiwu.torch_plugin.qvae.F.mse_loss", _spy_mse(seen)):
                loss = qvae.loss(targets, recon_x, posterior=None)

        self.assertEqual(seen["pred"], META)
        self.assertEqual(seen["target"], META)
        self.assertEqual(loss.device, META)
        self.assertEqual(targets.device, CPU)
        self.assertEqual(recon_x.device, META)

    def test_same_device_alignment_keeps_autograd(self):
        """Aligning tensors already on the parameter device must not detach."""
        qvae = _build_qvae("mse")
        qvae.train()
        inputs = torch.randn(2, INPUT_DIM, requires_grad=True)
        recon_x, posterior, _, _ = qvae(inputs)

        with _patch_aux_losses(qvae, CPU, inputs.shape[0]):
            loss = qvae.loss(inputs, recon_x, posterior)
        loss.backward()

        self.assertEqual(loss.device, CPU)
        self.assertIsNotNone(inputs.grad)
        self.assertEqual(inputs.grad.device, CPU)
        self.assertEqual(inputs.grad.shape, inputs.shape)
        self.assertIsNotNone(qvae.encoder.fc.weight.grad)
        self.assertEqual(qvae.encoder.fc.weight.grad.device, CPU)
        self.assertIsNotNone(qvae.decoder.fc.weight.grad)
        self.assertEqual(qvae.decoder.fc.weight.grad.device, CPU)
        self.assertTrue(torch.isfinite(qvae.encoder.fc.weight.grad).all().item())
        self.assertTrue(torch.isfinite(qvae.decoder.fc.weight.grad).all().item())


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestQvaeReconstructionParameterDeviceCuda(unittest.TestCase):
    """Accelerator coverage for parent moves between CPU and CUDA."""

    def test_cpu_batch_after_parent_move_to_cuda(self):
        """A CPU batch must reconstruct after ``parent.to('cuda')``."""
        parent = nn.Module()
        parent.qvae = _build_qvae("mse")
        parent.qvae.eval()
        parent.to("cuda")
        inputs = torch.randn(2, INPUT_DIM)

        with _deterministic_posterior():
            recon_x, _, logits, _ = parent.qvae(inputs)

        _assert_child_to_not_called(self, parent.qvae)
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(recon_x.device.type, "cuda")
        self.assertEqual(logits.device.type, "cuda")
        torch.testing.assert_close(
            recon_x.detach().cpu(), _cpu_reference_recon(parent.qvae, inputs)
        )

    def test_cuda_batch_after_parent_move_back_to_cpu(self):
        """Parameters moved CUDA then CPU must accept a CUDA batch."""
        parent = nn.Module()
        parent.qvae = _build_qvae("bernoulli")
        parent.qvae.eval()
        parent.qvae.set_train_bias(0.25)
        parent.to("cuda")
        parent.to("cpu")
        inputs = torch.zeros(2, INPUT_DIM, device="cuda")

        with _deterministic_posterior():
            recon_x, _, _, _ = parent.qvae(inputs)

        _assert_child_to_not_called(self, parent.qvae)
        self.assertEqual(inputs.device.type, "cuda")
        self.assertEqual(next(parent.qvae.encoder.parameters()).device, CPU)
        self.assertEqual(parent.qvae._train_bias.device, CPU)
        self.assertEqual(recon_x.device, CPU)
        self.assertEqual(recon_x.shape, inputs.shape)
        torch.testing.assert_close(recon_x, _cpu_reference_recon(parent.qvae, inputs))

    def test_construct_on_cuda_keeps_cpu_bias_and_matches_reference(self):
        """CUDA modules with a CPU bias buffer must still reconstruct.

        Building the encoder and decoder on CUDA does not move QVAE's
        ``_train_bias`` buffer. The buffer device must not win.
        """
        qvae = _build_qvae("bernoulli")
        qvae.eval()
        qvae.set_train_bias(0.25)
        qvae.encoder.to("cuda")
        qvae.decoder.to("cuda")
        qvae.encoder.direct_to_calls.clear()
        qvae.decoder.direct_to_calls.clear()
        inputs = torch.randn(2, INPUT_DIM)

        with _deterministic_posterior():
            recon_x, _, logits, _ = qvae(inputs)

        self.assertEqual(qvae.encoder.direct_to_calls, [])
        self.assertEqual(qvae.decoder.direct_to_calls, [])
        self.assertEqual(qvae._train_bias.device, CPU)
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(next(qvae.encoder.parameters()).device.type, "cuda")
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(recon_x.device.type, "cuda")
        self.assertEqual(recon_x.shape, inputs.shape)
        torch.testing.assert_close(
            recon_x.detach().cpu(), _cpu_reference_recon(qvae, inputs)
        )

    def test_encoder_cuda_decoder_cpu_splits_activations(self):
        """Encoder and decoder devices are followed independently."""
        qvae = _build_qvae("mse")
        qvae.eval()
        qvae.encoder.to("cuda")
        qvae.encoder.direct_to_calls.clear()
        inputs = torch.randn(2, INPUT_DIM)

        with _deterministic_posterior():
            recon_x, _, logits, zeta = qvae(inputs)

        self.assertEqual(qvae.encoder.direct_to_calls, [])
        self.assertEqual(qvae.decoder.direct_to_calls, [])
        self.assertEqual(inputs.device, CPU)
        self.assertEqual(next(qvae.encoder.parameters()).device.type, "cuda")
        self.assertEqual(next(qvae.decoder.parameters()).device, CPU)
        self.assertEqual(qvae.encoder.seen_devices[-1].type, "cuda")
        self.assertEqual(qvae.decoder.seen_devices[-1], CPU)
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(zeta.device.type, "cuda")
        self.assertEqual(recon_x.device, CPU)
        torch.testing.assert_close(recon_x, _cpu_reference_recon(qvae, inputs))

    def test_loss_aligns_cpu_targets_to_cuda_decoder(self):
        """``loss`` must score CPU inputs on CUDA decoder parameters."""
        qvae = _build_qvae("mse")
        qvae.eval()
        qvae.decoder.to("cuda")
        inputs = torch.randn(2, INPUT_DIM)
        with _deterministic_posterior():
            recon_x, posterior, _, _ = qvae(inputs)
        targets_before = inputs.detach().clone()

        with _patch_aux_losses(qvae, recon_x.device, inputs.shape[0]):
            loss = qvae.loss(inputs, recon_x, posterior)

        self.assertEqual(inputs.device, CPU)
        torch.testing.assert_close(inputs, targets_before)
        self.assertEqual(recon_x.device.type, "cuda")
        self.assertEqual(loss.device.type, "cuda")
        self.assertEqual(loss.shape, ())
        expected = torch.nn.functional.mse_loss(
            recon_x.detach().cpu(), inputs.detach().cpu(), reduction="sum"
        ) / inputs.size(0)
        torch.testing.assert_close(loss.detach().cpu(), expected)

    def test_backward_from_cpu_batch_reaches_cuda_parameters(self):
        """A device copy inside forward and loss must stay in the autograd graph."""
        qvae = _build_qvae("mse")
        qvae.train()
        qvae.to("cuda")
        inputs = torch.randn(2, INPUT_DIM, requires_grad=True)
        recon_x, posterior, _, _ = qvae(inputs)

        with _patch_aux_losses(qvae, recon_x.device, inputs.shape[0]):
            loss = qvae.loss(inputs, recon_x, posterior)
        loss.backward()

        self.assertEqual(loss.device.type, "cuda")
        self.assertIsNotNone(inputs.grad)
        self.assertEqual(inputs.grad.device, CPU)
        self.assertEqual(inputs.grad.shape, inputs.shape)
        self.assertIsNotNone(qvae.encoder.fc.weight.grad)
        self.assertEqual(qvae.encoder.fc.weight.grad.device.type, "cuda")
        self.assertIsNotNone(qvae.decoder.fc.weight.grad)
        self.assertEqual(qvae.decoder.fc.weight.grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(qvae.encoder.fc.weight.grad).all().item())
        self.assertTrue(torch.isfinite(qvae.decoder.fc.weight.grad).all().item())

    def test_parameter_device_follows_later_parent_moves(self):
        """Parent moves between CUDA and CPU must be reread, not cached."""
        parent = nn.Module()
        parent.qvae = _build_qvae("mse")
        parent.qvae.eval()
        cpu_batch = torch.randn(2, INPUT_DIM)

        parent.to("cuda")
        self.assertEqual(
            parent.qvae._parameter_device(parent.qvae.encoder, cpu_batch).type,
            "cuda",
        )
        cuda_recon, _, _, _ = parent.qvae(cpu_batch)
        _assert_child_to_not_called(self, parent.qvae)
        self.assertEqual(cpu_batch.device, CPU)
        self.assertEqual(cuda_recon.device.type, "cuda")

        parent.to("cpu")
        cuda_batch = torch.randn(2, INPUT_DIM, device="cuda")
        cpu_recon, _, _, _ = parent.qvae(cuda_batch)
        _assert_child_to_not_called(self, parent.qvae)
        self.assertEqual(
            parent.qvae._parameter_device(parent.qvae.decoder, cuda_batch), CPU
        )
        self.assertEqual(cuda_batch.device.type, "cuda")
        self.assertEqual(cpu_recon.device, CPU)

        parent.to("cuda")
        again, _, _, _ = parent.qvae(cpu_batch)
        _assert_child_to_not_called(self, parent.qvae)
        self.assertEqual(
            parent.qvae._parameter_device(parent.qvae.encoder, cpu_batch).type,
            "cuda",
        )
        self.assertEqual(again.device.type, "cuda")
        self.assertEqual(again.shape, cpu_batch.shape)
