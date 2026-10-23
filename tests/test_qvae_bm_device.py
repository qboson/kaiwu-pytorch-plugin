"""Regression tests for QVAE / Boltzmann-machine device alignment.

The encoder, decoder, and BM are sibling submodules. Moving only the BM
must not leave latent tensors on the encoder device. That move may be a
direct ``BM.to`` or a parent ``_apply``, which never calls the child's
``.to`` override (merged #164). ``_bm_parameter_device()`` reads the live
parameter device, so a stale ``bm.device`` cache is not the work device.

CPU cases use ``meta``, the same stand-in as ``tests/test_bm_device_sync.py``.
CUDA cases are skipped unless a real accelerator is present. They cover
construct-on-CUDA and parent move-to-CPU.

Open #1268 owns the KL ``entropy.to(...)`` alignment. This file does not
call ``_kl_dist_from`` across a device split. Open #1347 owns full-BM
``forward`` coercion, and open #1346 owns the MNIST trainer placement.
"""

# Parent-initiated moves go through ``_apply`` (see #164).
# pylint: disable=protected-access

import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin.qvae import QVAE
from kaiwu.torch_plugin.restricted_boltzmann_machine import (
    RestrictedBoltzmannMachine,
)

INPUT_DIM = 8
LATENT_DIM = 4
NUM_VISIBLE = LATENT_DIM // 2
NUM_HIDDEN = LATENT_DIM - NUM_VISIBLE
META = torch.device("meta")
CPU = torch.device("cpu")


class DummyEncoder(torch.nn.Module):
    """Return deterministic latent logits for QVAE tests."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.inputs = []

    def forward(self, features):
        """Record ``features`` and return ones on the same device."""
        self.inputs.append(features.detach().clone())
        return torch.ones(
            features.size(0),
            self.latent_dim,
            device=features.device,
            dtype=features.dtype,
        )


class DummyDecoder(torch.nn.Module):
    """Return zero reconstruction logits with the requested input dimension."""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, latent):
        """Return zeros matching ``latent``'s batch and device."""
        return torch.zeros(
            latent.size(0),
            self.input_dim,
            device=latent.device,
            dtype=latent.dtype,
        )


class DummySampler:  # pylint: disable=too-few-public-methods
    """Return a deterministic set of binary BM samples."""

    def solve(self, ising_matrix):
        """Return two all-zero samples shaped like ``ising_matrix``."""
        return np.zeros((2, ising_matrix.shape[0]), dtype=np.float32)


class DummyQVAE(QVAE):
    """QVAE test subclass using the supplied components."""

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
        """Return the sampler supplied to the constructor."""
        del sampler_type
        return self.sampler


class RecordingRBM(RestrictedBoltzmannMachine):
    """RestrictedBoltzmannMachine that records every direct ``.to()`` call."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direct_to_calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class ParameterFreeBM(torch.nn.Module):
    """Stand-in with no parameters, covering the device fallback."""

    def forward(self, states):
        """Reduce each row so callers still receive one value per sample."""
        return states.sum(dim=-1)


def _config():
    """Build the small QVAE config shared by these tests."""
    return types.SimpleNamespace(
        num_latent_units=LATENT_DIM,
        loss_type="mse",
        dist_beta=1.0,
        kl_beta=1e-6,
        weight_decay=0.01,
    )


def _build_qvae(device="cpu"):
    """Construct a QVAE whose BM records direct ``.to()`` calls.

    ``device=None`` leaves construction to ``RestrictedBoltzmannMachine``,
    which selects CUDA when it is available.
    """
    encoder = DummyEncoder(LATENT_DIM)
    decoder = DummyDecoder(INPUT_DIM)
    bm_kwargs = {}
    if device is not None:
        bm_kwargs["device"] = device
    return DummyQVAE(
        input_dimension=INPUT_DIM,
        activation_fct=None,
        config=_config(),
        encoder=encoder,
        decoder=decoder,
        bm=RecordingRBM(NUM_VISIBLE, NUM_HIDDEN, **bm_kwargs),
        sampler=DummySampler(),
    )


def _parent_move(qvae, device):
    """Move ``qvae`` the way a parent module does.

    ``nn.Module.to`` recurses through ``_apply`` and does not call the
    child's ``.to`` override.
    """
    parent = nn.Module()
    parent.qvae = qvae
    parent.to(device)
    return parent


def _force_cached_device(bm, device):
    """Overwrite the plain ``bm.device`` attribute without moving parameters."""
    bm.device = device


def _negative_samples(device, rows=2):
    """Deterministic model samples already on ``device``."""
    return torch.zeros(rows, LATENT_DIM, device=device)


def _retarget_storage(device):
    """Return an ``_apply`` function that places empty storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``BM.to``.
    """

    def _apply_fn(tensor):
        if tensor.device == device:
            return tensor
        return torch.empty(tensor.shape, dtype=tensor.dtype, device=device)

    return _apply_fn


def _cpu_reference_energy(qvae, features):
    """Energy of ``features`` against a CPU copy of the BM weights."""
    reference = RestrictedBoltzmannMachine(
        NUM_VISIBLE,
        NUM_HIDDEN,
        quadratic_coef=qvae.bm.quadratic_coef.detach().to(device="cpu").clone(),
        linear_bias=qvae.bm.linear_bias.detach().to(device="cpu").clone(),
        device="cpu",
    )
    encoded = qvae.encoder(features.detach().to(device="cpu"))
    visible = (encoded > 0).to(dtype=encoded.dtype).float()
    return reference(visible)


class TestQVaeBmDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def setUp(self):
        self.qvae = _build_qvae(device="cpu")

    def test_energy_after_bm_only_move(self):
        """``energy()`` must accept encoder outputs when only the BM moved."""
        self.qvae.bm.to(META)
        features = torch.randn(2, INPUT_DIM)

        energy = self.qvae.energy(features)

        self.assertEqual(features.device, CPU)
        self.assertEqual(self.qvae.encoder.inputs[-1].device, CPU)
        self.assertEqual(energy.shape, (features.size(0),))
        self.assertEqual(energy.device, META)

    def test_bm_loss_after_bm_only_move(self):
        """``bm_loss()`` must align detached logits with the BM device."""
        self.qvae.bm.to(META)
        logits = torch.ones(2, LATENT_DIM)
        negative = _negative_samples(META)
        with mock.patch.object(self.qvae.bm, "sample", return_value=negative):
            loss = self.qvae.bm_loss(logits, bm_weight_decay=0.01)

        self.assertEqual(logits.device, CPU)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.device, META)

    def test_kl_cross_entropy_after_bm_only_move(self):
        """KL cross-entropy must score BM energies on the BM parameter device."""
        self.qvae.bm.to(META)
        logit_q = torch.zeros(2, LATENT_DIM)
        negative = _negative_samples(META)
        with mock.patch.object(self.qvae.bm, "sample", return_value=negative):
            cross_entropy = self.qvae._cross_entropy(logit_q)

        self.assertEqual(logit_q.device, CPU)
        self.assertEqual(cross_entropy.device, META)

    def test_parent_move_does_not_call_bm_to(self):
        """A parent ``.to()`` updates BM parameters without calling ``BM.to``.

        The test encoder has no parameters, so its outputs stay on the batch
        device while the BM weights move. That is the sibling split.
        """
        _parent_move(self.qvae, META)
        features = torch.randn(2, INPUT_DIM)

        energy = self.qvae.energy(features)

        self.assertEqual(self.qvae.bm.direct_to_calls, [])
        self.assertEqual(self.qvae.bm.device, META)
        self.assertEqual(self.qvae.bm.quadratic_coef.device, META)
        self.assertEqual(self.qvae._bm_parameter_device(), META)
        self.assertEqual(features.device, CPU)
        self.assertEqual(self.qvae.encoder.inputs[-1].device, CPU)
        self.assertEqual(energy.shape, (features.size(0),))
        self.assertEqual(energy.device, META)

    def test_bm_loss_after_parent_move(self):
        """``bm_loss()`` follows a parent move that never called ``BM.to``."""
        _parent_move(self.qvae, META)
        logits = torch.ones(2, LATENT_DIM)
        negative = _negative_samples(META)
        with mock.patch.object(self.qvae.bm, "sample", return_value=negative):
            loss = self.qvae.bm_loss(logits, bm_weight_decay=0.01)

        self.assertEqual(self.qvae.bm.direct_to_calls, [])
        self.assertEqual(logits.device, CPU)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.device, META)

    def test_cross_entropy_after_parent_move(self):
        """Cross-entropy follows a parent move that never called ``BM.to``."""
        _parent_move(self.qvae, META)
        logit_q = torch.zeros(2, LATENT_DIM)
        negative = _negative_samples(META)
        with mock.patch.object(self.qvae.bm, "sample", return_value=negative):
            cross_entropy = self.qvae._cross_entropy(logit_q)

        self.assertEqual(self.qvae.bm.direct_to_calls, [])
        self.assertEqual(logit_q.device, CPU)
        self.assertEqual(cross_entropy.device, META)

    def test_energy_follows_parameters_when_device_cache_is_stale(self):
        """A manually stale ``bm.device`` must not choose the energy device."""
        self.qvae.bm._apply(lambda tensor: tensor.to(META))
        _force_cached_device(self.qvae.bm, CPU)

        features = torch.randn(2, INPUT_DIM)
        energy = self.qvae.energy(features)

        self.assertEqual(self.qvae.bm.direct_to_calls, [])
        self.assertEqual(self.qvae.bm.device, CPU)
        self.assertEqual(self.qvae._bm_parameter_device(), META)
        self.assertEqual(features.device, CPU)
        self.assertEqual(energy.shape, (features.size(0),))
        self.assertEqual(energy.device, META)

    def test_losses_follow_parameters_when_device_cache_is_stale(self):
        """BM loss and cross-entropy must ignore a stale ``bm.device`` cache.

        ``sample()`` still builds its buffers from ``bm.device`` (open
        #1268). Negative samples are supplied on the parameter device so
        this test only checks the tensors QVAE aligns.
        """
        self.qvae.bm._apply(lambda tensor: tensor.to(META))
        _force_cached_device(self.qvae.bm, CPU)
        negative = _negative_samples(META)
        logits = torch.ones(2, LATENT_DIM)
        logit_q = torch.zeros(2, LATENT_DIM)
        with mock.patch.object(self.qvae.bm, "sample", return_value=negative):
            loss = self.qvae.bm_loss(logits, bm_weight_decay=0.01)
            cross_entropy = self.qvae._cross_entropy(logit_q)

        self.assertEqual(self.qvae.bm.device, CPU)
        self.assertEqual(self.qvae._bm_parameter_device(), META)
        self.assertEqual(logits.device, CPU)
        self.assertEqual(logit_q.device, CPU)
        self.assertEqual(loss.device, META)
        self.assertEqual(cross_entropy.device, META)

    def test_parameter_device_follows_a_later_move(self):
        """The helper must be reread after each move, not cached on first use."""
        self.assertEqual(self.qvae._bm_parameter_device(), CPU)
        first = self.qvae.energy(torch.randn(2, INPUT_DIM))
        self.assertEqual(first.device, CPU)

        _parent_move(self.qvae, META)
        second = self.qvae.energy(torch.randn(2, INPUT_DIM))
        self.assertEqual(self.qvae.bm.direct_to_calls, [])
        self.assertEqual(self.qvae._bm_parameter_device(), META)
        self.assertEqual(second.device, META)

        # ``.to(cpu)`` cannot copy out of meta; ``_apply`` still retargets.
        self.qvae.bm._apply(_retarget_storage(CPU))
        third = self.qvae.energy(torch.randn(2, INPUT_DIM))
        self.assertEqual(self.qvae.bm.direct_to_calls, [])
        self.assertEqual(self.qvae._bm_parameter_device(), CPU)
        self.assertEqual(third.device, CPU)

    def test_dtype_only_move_keeps_parameter_device(self):
        """``.to(dtype=...)`` must not change the device the helper reports."""
        self.qvae.bm.to(dtype=torch.float64)

        self.assertEqual(self.qvae.bm.quadratic_coef.dtype, torch.float64)
        self.assertEqual(self.qvae.bm.linear_bias.dtype, torch.float64)
        self.assertEqual(self.qvae.bm.device, CPU)
        self.assertEqual(self.qvae._bm_parameter_device(), CPU)

    def test_parameter_device_falls_back_to_sibling_parameter(self):
        """With no BM parameters, report another QVAE parameter's device."""
        anchor = torch.nn.Parameter(torch.zeros(1, device=META))
        self.qvae.encoder.register_parameter("anchor", anchor)
        self.qvae.bm = ParameterFreeBM()

        self.assertEqual(self.qvae._bm_parameter_device(), META)
        self.assertEqual(list(self.qvae.bm.parameters()), [])

    def test_parameter_device_falls_back_to_cpu_without_parameters(self):
        """No BM parameters and no other parameters yields CPU, not an error."""
        self.qvae.bm = ParameterFreeBM()

        self.assertEqual(self.qvae._bm_parameter_device(), CPU)
        self.assertIsNone(next(self.qvae.parameters(), None))

    def test_cross_entropy_keeps_grad_through_same_device_alignment(self):
        """Aligning logits already on the BM device must not drop autograd."""
        logit_q = torch.zeros(2, LATENT_DIM, requires_grad=True)
        negative = _negative_samples(CPU)
        with mock.patch.object(self.qvae.bm, "sample", return_value=negative):
            cross_entropy = self.qvae._cross_entropy(logit_q)
        cross_entropy.backward()

        self.assertEqual(cross_entropy.device, CPU)
        self.assertIsNotNone(logit_q.grad)
        self.assertEqual(logit_q.grad.device, CPU)
        self.assertEqual(logit_q.grad.shape, logit_q.shape)
        self.assertIsNotNone(self.qvae.bm.quadratic_coef.grad)
        self.assertEqual(self.qvae.bm.quadratic_coef.grad.device, CPU)
        self.assertTrue(torch.isfinite(self.qvae.bm.quadratic_coef.grad).all())


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestQVaeBmDeviceAcrossAccelerators(unittest.TestCase):
    """Construct-on-CUDA and parent move-to-CPU coverage for QVAE."""

    def setUp(self):
        self.qvae = _build_qvae(device=None)

    def test_construct_on_cuda_accepts_cpu_batch(self):
        """CPU encoder outputs must score on CUDA BM parameters."""
        self.assertEqual(
            self.qvae.bm.quadratic_coef.device.type,
            "cuda",
            "construction should auto-select CUDA",
        )
        features = torch.randn(2, INPUT_DIM)
        energy = self.qvae.energy(features)
        logits = torch.ones(2, LATENT_DIM)
        loss = self.qvae.bm_loss(logits)
        logit_q = torch.zeros(2, LATENT_DIM)
        cross_entropy = self.qvae._cross_entropy(logit_q)

        self.assertEqual(features.device, CPU)
        self.assertEqual(logits.device, CPU)
        self.assertEqual(logit_q.device, CPU)
        self.assertEqual(energy.shape, (2,))
        self.assertEqual(energy.device.type, "cuda")
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.device.type, "cuda")
        self.assertEqual(cross_entropy.device.type, "cuda")
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(cross_entropy).all())
        torch.testing.assert_close(
            energy.detach().cpu(),
            _cpu_reference_energy(self.qvae, features),
        )

    def test_parent_move_to_cpu_accepts_cuda_batch(self):
        """After ``parent.to('cpu')``, a CUDA encoder batch must score on CPU."""
        parent = nn.Module()
        parent.qvae = self.qvae
        self.assertEqual(self.qvae.bm.quadratic_coef.device.type, "cuda")

        parent.to("cpu")
        features = torch.randn(2, INPUT_DIM, device="cuda")
        energy = self.qvae.energy(features)

        self.assertEqual(self.qvae.bm.direct_to_calls, [])
        self.assertEqual(self.qvae.bm.quadratic_coef.device.type, "cpu")
        self.assertEqual(self.qvae.bm.device, CPU)
        self.assertEqual(features.device.type, "cuda")
        self.assertEqual(self.qvae.encoder.inputs[-1].device.type, "cuda")
        self.assertEqual(energy.device, CPU)
        self.assertEqual(energy.shape, (2,))
        torch.testing.assert_close(energy, _cpu_reference_energy(self.qvae, features))

    def test_stale_device_cache_still_scores_on_cuda(self):
        """QVAE must score on CUDA parameters when ``bm.device`` says CPU.

        ``sample()`` still builds its buffers from ``bm.device`` (open
        #1268). Negative samples are supplied on the parameter device so
        this test only checks the tensors QVAE aligns.
        """
        _force_cached_device(self.qvae.bm, CPU)
        features = torch.randn(2, INPUT_DIM)
        energy = self.qvae.energy(features)
        negative = _negative_samples(self.qvae._bm_parameter_device())
        with mock.patch.object(self.qvae.bm, "sample", return_value=negative):
            loss = self.qvae.bm_loss(torch.ones(2, LATENT_DIM))
            cross_entropy = self.qvae._cross_entropy(torch.zeros(2, LATENT_DIM))

        self.assertEqual(self.qvae.bm.device, CPU)
        self.assertEqual(self.qvae._bm_parameter_device().type, "cuda")
        self.assertEqual(energy.device.type, "cuda")
        self.assertEqual(energy.shape, (2,))
        self.assertEqual(loss.device.type, "cuda")
        self.assertEqual(cross_entropy.device.type, "cuda")
        torch.testing.assert_close(
            energy.detach().cpu(),
            _cpu_reference_energy(self.qvae, features),
        )

    def test_parameter_device_follows_later_parent_moves(self):
        """Parent moves between CUDA and CPU must be reread, not cached."""
        self.assertEqual(self.qvae._bm_parameter_device().type, "cuda")
        parent = nn.Module()
        parent.qvae = self.qvae

        parent.to("cpu")
        cpu_energy = self.qvae.energy(torch.randn(2, INPUT_DIM))
        self.assertEqual(self.qvae.bm.direct_to_calls, [])
        self.assertEqual(self.qvae._bm_parameter_device(), CPU)
        self.assertEqual(cpu_energy.device, CPU)

        parent.to("cuda")
        cuda_energy = self.qvae.energy(torch.randn(2, INPUT_DIM))
        self.assertEqual(self.qvae.bm.direct_to_calls, [])
        self.assertEqual(self.qvae._bm_parameter_device().type, "cuda")
        self.assertEqual(cuda_energy.device.type, "cuda")
        self.assertEqual(cuda_energy.shape, (2,))

    def test_backward_from_cpu_logits_reaches_cuda_parameters(self):
        """A device copy inside cross-entropy must stay in the autograd graph."""
        logit_q = torch.zeros(2, LATENT_DIM, requires_grad=True)

        cross_entropy = self.qvae._cross_entropy(logit_q)
        cross_entropy.backward()

        self.assertEqual(cross_entropy.device.type, "cuda")
        self.assertIsNotNone(logit_q.grad)
        self.assertEqual(logit_q.grad.device, CPU)
        self.assertEqual(logit_q.grad.shape, logit_q.shape)
        self.assertTrue(torch.isfinite(logit_q.grad).all())
        self.assertIsNotNone(self.qvae.bm.quadratic_coef.grad)
        self.assertEqual(self.qvae.bm.quadratic_coef.grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(self.qvae.bm.quadratic_coef.grad).all())
        self.assertIsNotNone(self.qvae.bm.linear_bias.grad)
        self.assertEqual(self.qvae.bm.linear_bias.grad.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
