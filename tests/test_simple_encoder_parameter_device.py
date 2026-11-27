"""Regression tests for ``SimpleEncoder.hierarchical_posterior`` device alignment.

``hierarchical_posterior`` encodes ``x`` with the encoder's linear weights and
then draws ``rho``. ``smoothing_distribution.icdf`` mixes ``rho`` with that
activation. A parent ``nn.Module._apply`` (the path ``nn.Module.to`` uses for
children; merged #164 / ``tests/test_bm_device_sync.py``) moves the weights
and does not call ``SimpleEncoder.to``. ``rho`` has to follow the live
parameter device, and the caller's storage has to stay put.

``.to(cpu)`` cannot copy out of a meta tensor, so a meta round trip retargets
storage with ``_apply`` and the next call rereads that device. CPU results
stay inside the unit interval. Meta tensors have no data, so those tests
check device, dtype, and storage rather than values.

Out of scope: the CPU hierarchy-count / shape contract (#1294; its ``icdf``
stub drops ``rho``), library ``QVAE.forward`` (#1370), cell ``_encode``
(#1371), and the MNIST feature-extractor / trainer placement fixes
(#1343 / #1346). This file does not import the Kaiwu SDK.
"""

# Parent moves go through ``_apply`` (see merged #164).
# pylint: disable=protected-access

import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ROOT / "example" / "qvae_mnist" / "model" / "networks.py"

_CPU = torch.device("cpu")
_META = torch.device("meta")
_SAMPLE_SHAPE = (2, 2)


def _load_networks():
    """Load the MNIST network module without importing the Kaiwu SDK."""
    spec = importlib.util.spec_from_file_location(
        "qvae_mnist_networks_device", NETWORKS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SimpleEncoder = _load_networks().SimpleEncoder


class _MixingSmoothing:  # pylint: disable=too-few-public-methods
    """Inverse CDF that consumes both ``rho`` and ``q`` and records them.

    The result stays inside ``[0, 1]`` when both inputs do, which is the
    range a smoothing quantile is supposed to produce.
    """

    def __init__(self):
        self.seen_rho = []
        self.seen_q = []

    def icdf(self, rho, q):
        """Mix the uniform sample with the encoded Bernoulli probability."""
        self.seen_rho.append(rho)
        self.seen_q.append(q)
        return rho * q + (1.0 - rho) * (1.0 - q)


class _RecordingEncoder(SimpleEncoder):  # pylint: disable=too-few-public-methods
    """Encoder that records direct ``.to()`` calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_to_calls = []

    def to(self, *args, **kwargs):
        """Record the call, then perform the real move."""
        self.direct_to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def _encoder(cls=SimpleEncoder, dtype=torch.float32):
    """Return a one-layer encoder of type ``cls``."""
    encoder = cls(
        smoothing_distribution=_MixingSmoothing(),
        node_sequence=[[3, 2]],
        activation_fct=torch.sigmoid,
    )
    if dtype != torch.float32:
        encoder.to(dtype=dtype)
    return encoder


def _parameter_free_encoder():
    """Return an encoder with no linear layers and therefore no parameters."""
    return SimpleEncoder(smoothing_distribution=_MixingSmoothing())


def _features(dtype=torch.float32, device=_CPU):
    """Return a noncontiguous ``(2, 3)`` batch with distinct values."""
    base = torch.arange(24, dtype=dtype, device=device).reshape(2, 3, 4)
    return base[:, :, 0]


def _move_parameters(module, device):
    """Move parameters the way a parent does: ``_apply``, not ``.to``."""
    module._apply(lambda tensor: tensor.to(device))


def _retarget_storage(device):
    """Return an ``_apply`` callback that places empty storage on ``device``.

    ``Module.to`` cannot copy out of ``meta``. Replacing storage still goes
    through ``_apply``, so the parameter device changes without ``.to()``.
    """

    def _apply_fn(tensor):
        """Place ``tensor`` on ``device`` without reading its data."""
        if tensor.device == device:
            return tensor
        return torch.empty(tensor.shape, dtype=tensor.dtype, device=device)

    return _apply_fn


def _samples(encoder, features):
    """Return posterior samples, keeping only this call's ``rho`` records."""
    smoother = encoder.smoothing_distribution
    smoother.seen_rho.clear()
    smoother.seen_q.clear()
    _ignored, samples = encoder.hierarchical_posterior(features)
    return samples


def _assert_unchanged(case, tensor, before, pointer, values=True):
    """Assert ``tensor`` kept its storage, device, dtype, and layout.

    ``values`` is false for meta tensors, which have no readable data.
    """
    case.assertEqual(tensor.data_ptr(), pointer)
    case.assertEqual(tensor.device, before.device)
    case.assertEqual(tensor.dtype, before.dtype)
    case.assertFalse(tensor.is_contiguous())
    if values:
        case.assertTrue(torch.equal(tensor, before))


def _assert_rho(case, encoder, device, dtype):
    """Every hierarchy level must have drawn ``rho`` on ``device``."""
    rhos = encoder.smoothing_distribution.seen_rho
    qs = encoder.smoothing_distribution.seen_q
    case.assertEqual(len(rhos), encoder.num_latent_hierarchy_levels)
    case.assertEqual(len(qs), len(rhos))
    for rho, q in zip(rhos, qs):
        case.assertEqual(rho.device, device)
        case.assertEqual(q.device, device)
        case.assertEqual(rho.dtype, dtype)
        case.assertEqual(q.dtype, dtype)
        case.assertEqual(rho.shape, q.shape)


def _assert_samples(case, samples, device, dtype, shape):
    """Assert every sample's device, dtype, and shape."""
    case.assertGreater(len(samples), 0)
    for sample in samples:
        case.assertEqual(sample.device, device)
        case.assertEqual(sample.dtype, dtype)
        case.assertEqual(sample.shape, shape)


def _assert_unit_interval(case, samples):
    """CPU / CUDA samples are finite and lie in ``[0, 1]``."""
    for sample in samples:
        case.assertTrue(torch.isfinite(sample).all().item())
        case.assertTrue(torch.all((sample >= 0.0) & (sample <= 1.0)).item())


def _assert_cannot_copy_out_of_meta(case):
    """``.to(cpu)`` cannot materialise a meta tensor or a meta module."""
    meta_value = torch.empty(1, device=_META)
    with case.assertRaises(NotImplementedError) as caught:
        meta_value.to(_CPU)
    case.assertIn("meta", str(caught.exception).lower())

    module = nn.Linear(2, 2)
    _move_parameters(module, _META)
    with case.assertRaises(NotImplementedError) as caught:
        module.to(_CPU)
    case.assertIn("meta", str(caught.exception).lower())
    case.assertEqual(next(module.parameters()).device, _META)


class TestSimpleEncoderPosteriorParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_cpu_posterior_stays_in_unit_interval(self):
        """An unmoved CPU encoder returns one finite unit-interval sample per level."""
        encoder = _encoder()
        features = torch.zeros(2, 3)
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        self.assertEqual(features.data_ptr(), pointer)
        self.assertTrue(torch.equal(features, before))
        self.assertEqual(encoder._parameter_device(features.device), _CPU)
        _assert_rho(self, encoder, _CPU, torch.float32)
        _assert_samples(self, samples, _CPU, torch.float32, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)

    def test_parent_to_meta_skips_encoder_to(self):
        """``nn.Module.to`` must move the encoder without calling its ``.to``."""
        encoder = _encoder(_RecordingEncoder)
        parent = nn.Module()
        parent.encoder = encoder

        parent.to(_META)

        self.assertEqual(encoder.direct_to_calls, [])
        self.assertEqual(next(encoder.parameters()).device, _META)

    def test_parent_apply_accepts_cpu_batch(self):
        """CPU features must score on meta, and their storage must stay put."""
        encoder = _encoder(_RecordingEncoder)
        parent = nn.Module()
        parent.encoder = encoder
        _move_parameters(parent, _META)
        features = _features()
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        self.assertEqual(encoder.direct_to_calls, [])
        _assert_unchanged(self, features, before, pointer)
        self.assertEqual(encoder._parameter_device(features.device), _META)
        self.assertEqual(next(encoder.parameters()).device, _META)
        _assert_rho(self, encoder, _META, torch.float32)
        _assert_samples(self, samples, _META, torch.float32, _SAMPLE_SHAPE)

    def test_rho_follows_parameters_when_input_is_already_there(self):
        """``rho`` follows the weights when ``x`` is already on that device."""
        encoder = _encoder()
        _move_parameters(encoder, _META)
        features = _features(device=_META)
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        _assert_unchanged(self, features, before, pointer, values=False)
        self.assertEqual(encoder._parameter_device(features.device), _META)
        _assert_rho(self, encoder, _META, torch.float32)
        _assert_samples(self, samples, _META, torch.float32, _SAMPLE_SHAPE)
        for rho in encoder.smoothing_distribution.seen_rho:
            self.assertIsNot(rho, features)

    def test_float64_parameters_keep_feature_dtype_and_storage(self):
        """Moving float64 weights must not recast or rewrite the caller features."""
        encoder = _encoder(dtype=torch.float64)
        parent = nn.Module()
        parent.encoder = encoder
        _move_parameters(parent, _META)
        features = _features(dtype=torch.float64)
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        _assert_unchanged(self, features, before, pointer)
        weight = next(encoder.parameters())
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device, _META)
        _assert_rho(self, encoder, _META, torch.float64)
        _assert_samples(self, samples, _META, torch.float64, _SAMPLE_SHAPE)

    def test_dtype_only_move_keeps_cpu_device(self):
        """``.to(dtype=...)`` must keep CPU and must not call the child ``.to``."""
        encoder = _encoder(_RecordingEncoder)
        parent = nn.Module()
        parent.encoder = encoder
        parent.to(dtype=torch.float64)
        features = _features(dtype=torch.float64)
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        self.assertEqual(encoder.direct_to_calls, [])
        _assert_unchanged(self, features, before, pointer)
        weight = next(encoder.parameters())
        self.assertEqual(weight.dtype, torch.float64)
        self.assertEqual(weight.device, _CPU)
        self.assertEqual(encoder._parameter_device(features.device), _CPU)
        _assert_rho(self, encoder, _CPU, torch.float64)
        _assert_samples(self, samples, _CPU, torch.float64, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)

    def test_direct_dtype_only_posterior_stays_in_unit_interval(self):
        """A direct float64 cast stays on CPU and leaves the caller features put."""
        encoder = _encoder()
        encoder.to(dtype=torch.float64)
        features = _features(dtype=torch.float64)
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        _assert_unchanged(self, features, before, pointer)
        self.assertEqual(next(encoder.parameters()).dtype, torch.float64)
        self.assertEqual(next(encoder.parameters()).device, _CPU)
        _assert_rho(self, encoder, _CPU, torch.float64)
        _assert_samples(self, samples, _CPU, torch.float64, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)

    def test_parameter_device_is_reread_after_meta_round_trip(self):
        """The helper must be reread after each move, not cached.

        ``.to(cpu)`` cannot copy out of meta. ``_apply`` still retargets
        storage, and the next posterior follows that new device.
        """
        _assert_cannot_copy_out_of_meta(self)
        encoder = _encoder(_RecordingEncoder)
        parent = nn.Module()
        parent.encoder = encoder
        features = _features()
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)
        self.assertEqual(encoder._parameter_device(features.device), _CPU)
        _assert_samples(self, samples, _CPU, torch.float32, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)

        _move_parameters(parent, _META)
        self.assertEqual(encoder._parameter_device(features.device), _META)
        samples = _samples(encoder, features)
        self.assertEqual(encoder.direct_to_calls, [])
        _assert_rho(self, encoder, _META, torch.float32)
        _assert_samples(self, samples, _META, torch.float32, _SAMPLE_SHAPE)

        with self.assertRaises(NotImplementedError) as caught:
            parent.to(_CPU)
        self.assertIn("meta", str(caught.exception).lower())
        self.assertEqual(encoder.direct_to_calls, [])
        self.assertEqual(next(encoder.parameters()).device, _META)

        parent._apply(_retarget_storage(_CPU))
        self.assertEqual(encoder.direct_to_calls, [])
        self.assertEqual(encoder._parameter_device(features.device), _CPU)
        samples = _samples(encoder, features)
        self.assertEqual(next(encoder.parameters()).device, _CPU)
        _assert_rho(self, encoder, _CPU, torch.float32)
        _assert_samples(self, samples, _CPU, torch.float32, _SAMPLE_SHAPE)

        _move_parameters(parent, _META)
        samples = _samples(encoder, features)
        self.assertEqual(encoder.direct_to_calls, [])
        self.assertEqual(encoder._parameter_device(features.device), _META)
        _assert_rho(self, encoder, _META, torch.float32)
        _assert_samples(self, samples, _META, torch.float32, _SAMPLE_SHAPE)
        _assert_unchanged(self, features, before, pointer)

    def test_parameter_device_uses_live_parameters_or_fallback(self):
        """Empty encoders fall back; a later ``_apply`` wins over that fallback."""
        empty = _parameter_free_encoder()
        self.assertIsNone(next(empty.parameters(), None))
        self.assertEqual(empty._parameter_device(_CPU), _CPU)
        self.assertEqual(empty._parameter_device(_META), _META)

        encoder = _encoder()
        self.assertEqual(encoder._parameter_device(_META), _CPU)
        _move_parameters(encoder, _META)
        self.assertEqual(encoder._parameter_device(_CPU), _META)
        self.assertEqual(next(encoder.parameters()).device, _META)

    def test_parameter_free_encoder_follows_the_input_device(self):
        """With no weights, ``rho`` and the samples follow the caller activation."""
        encoder = _parameter_free_encoder()
        features = _features()
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        self.assertIsNone(next(encoder.parameters(), None))
        _assert_unchanged(self, features, before, pointer)
        self.assertEqual(encoder._parameter_device(features.device), _CPU)
        _assert_rho(self, encoder, _CPU, torch.float32)
        _assert_samples(self, samples, _CPU, torch.float32, features.shape)
        _assert_unit_interval(self, samples)

        meta_features = _features(device=_META)
        meta_before = meta_features.clone()
        meta_pointer = meta_features.data_ptr()
        samples = _samples(encoder, meta_features)
        _assert_unchanged(self, meta_features, meta_before, meta_pointer, values=False)
        self.assertEqual(encoder._parameter_device(meta_features.device), _META)
        _assert_rho(self, encoder, _META, torch.float32)
        _assert_samples(self, samples, _META, torch.float32, meta_features.shape)

    def test_same_device_alignment_keeps_autograd(self):
        """A same-device ``.to`` must still backpropagate into the batch and weights."""
        torch.manual_seed(0)
        encoder = _encoder()
        features = torch.tensor(
            [[0.2, -0.1, 0.3], [0.0, 0.4, -0.2]], requires_grad=True
        )

        samples = _samples(encoder, features)
        samples[0].sum().backward()

        weight = encoder._layers[0].weight
        self.assertEqual(samples[0].device, _CPU)
        self.assertIsNotNone(features.grad)
        self.assertEqual(features.grad.device, _CPU)
        self.assertEqual(features.grad.shape, features.shape)
        self.assertTrue(torch.isfinite(features.grad).all().item())
        self.assertIsNotNone(weight.grad)
        self.assertEqual(weight.grad.device, _CPU)
        self.assertTrue(torch.isfinite(weight.grad).all().item())
        self.assertGreater(float(weight.grad.abs().sum()), 0.0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSimpleEncoderPosteriorCudaParameterDevice(unittest.TestCase):
    """End-to-end coverage of a parent move onto CUDA. Skipped without a GPU."""

    def test_parent_apply_accepts_cpu_batch(self):
        """CPU features must yield CUDA samples after a parent ``_apply``."""
        encoder = _encoder(_RecordingEncoder)
        parent = nn.Module()
        parent.encoder = encoder
        cuda = torch.device("cuda")
        _move_parameters(parent, cuda)
        features = _features()
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        device = next(encoder.parameters()).device
        self.assertEqual(encoder.direct_to_calls, [])
        _assert_unchanged(self, features, before, pointer)
        self.assertEqual(device.type, "cuda")
        self.assertEqual(encoder._parameter_device(features.device), device)
        _assert_rho(self, encoder, device, torch.float32)
        _assert_samples(self, samples, device, torch.float32, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)

    def test_rho_follows_parameters_when_input_is_already_there(self):
        """``rho`` stays on CUDA when the batch is already on the weights."""
        encoder = _encoder()
        cuda = torch.device("cuda")
        _move_parameters(encoder, cuda)
        features = _features(device=cuda)
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        device = next(encoder.parameters()).device
        _assert_unchanged(self, features, before, pointer)
        _assert_rho(self, encoder, device, torch.float32)
        _assert_samples(self, samples, device, torch.float32, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)

    def test_device_is_reread_across_cuda_and_cpu(self):
        """CUDA and CPU parent moves must be reread, and must not call ``.to``."""
        encoder = _encoder(_RecordingEncoder)
        parent = nn.Module()
        parent.encoder = encoder
        features = _features()
        before = features.clone()
        pointer = features.data_ptr()
        cuda = torch.device("cuda")

        _move_parameters(parent, cuda)
        samples = _samples(encoder, features)
        device = next(encoder.parameters()).device
        self.assertEqual(encoder._parameter_device(features.device), device)
        self.assertEqual(device.type, "cuda")
        _assert_rho(self, encoder, device, torch.float32)
        _assert_samples(self, samples, device, torch.float32, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)

        _move_parameters(parent, _CPU)
        samples = _samples(encoder, features)
        self.assertEqual(encoder._parameter_device(features.device), _CPU)
        _assert_rho(self, encoder, _CPU, torch.float32)
        _assert_samples(self, samples, _CPU, torch.float32, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)

        _move_parameters(parent, cuda)
        samples = _samples(encoder, features)
        device = next(encoder.parameters()).device
        self.assertEqual(encoder.direct_to_calls, [])
        _assert_unchanged(self, features, before, pointer)
        _assert_rho(self, encoder, device, torch.float32)
        _assert_samples(self, samples, device, torch.float32, _SAMPLE_SHAPE)

    def test_float64_cuda_move_keeps_dtype(self):
        """A float64 CUDA move must not downcast the features or ``rho``."""
        encoder = _encoder(dtype=torch.float64)
        parent = nn.Module()
        parent.encoder = encoder
        _move_parameters(parent, torch.device("cuda"))
        features = _features(dtype=torch.float64)
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        device = next(encoder.parameters()).device
        _assert_unchanged(self, features, before, pointer)
        self.assertEqual(device.type, "cuda")
        self.assertEqual(next(encoder.parameters()).dtype, torch.float64)
        _assert_rho(self, encoder, device, torch.float64)
        _assert_samples(self, samples, device, torch.float64, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)

    def test_dtype_only_move_keeps_cuda_device(self):
        """``.to(dtype=...)`` on CUDA must keep CUDA and must not call child ``.to``."""
        encoder = _encoder(_RecordingEncoder)
        parent = nn.Module()
        parent.encoder = encoder
        _move_parameters(parent, torch.device("cuda"))
        parent.to(dtype=torch.float64)
        features = _features(dtype=torch.float64)
        before = features.clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        device = next(encoder.parameters()).device
        self.assertEqual(encoder.direct_to_calls, [])
        _assert_unchanged(self, features, before, pointer)
        self.assertEqual(device.type, "cuda")
        self.assertEqual(next(encoder.parameters()).dtype, torch.float64)
        _assert_rho(self, encoder, device, torch.float64)
        _assert_samples(self, samples, device, torch.float64, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)

    def test_backward_from_cpu_batch_reaches_cuda_weights(self):
        """The batch copy inside the posterior must stay in the autograd graph."""
        torch.manual_seed(0)
        encoder = _encoder()
        parent = nn.Module()
        parent.encoder = encoder
        _move_parameters(parent, torch.device("cuda"))
        features = torch.tensor(
            [[0.2, -0.1, 0.3], [0.0, 0.4, -0.2]], requires_grad=True
        )
        pointer = features.data_ptr()

        samples = _samples(encoder, features)
        samples[0].sum().backward()

        weight = encoder._layers[0].weight
        self.assertEqual(features.data_ptr(), pointer)
        self.assertEqual(features.device, _CPU)
        self.assertEqual(samples[0].device.type, "cuda")
        self.assertIsNotNone(features.grad)
        self.assertEqual(features.grad.device, _CPU)
        self.assertEqual(features.grad.shape, features.shape)
        self.assertTrue(torch.isfinite(features.grad).all().item())
        self.assertIsNotNone(weight.grad)
        self.assertEqual(weight.grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(weight.grad).all().item())
        self.assertGreater(float(weight.grad.abs().sum()), 0.0)

    def test_cuda_batch_follows_cpu_parameters(self):
        """A CUDA batch must still be scored on a CPU encoder."""
        encoder = _encoder()
        features = _features().to(device="cuda")
        before = features.detach().cpu().clone()
        pointer = features.data_ptr()

        samples = _samples(encoder, features)

        self.assertEqual(features.data_ptr(), pointer)
        self.assertEqual(features.device.type, "cuda")
        self.assertEqual(features.dtype, torch.float32)
        torch.testing.assert_close(features.detach().cpu(), before)
        self.assertEqual(next(encoder.parameters()).device, _CPU)
        _assert_rho(self, encoder, _CPU, torch.float32)
        _assert_samples(self, samples, _CPU, torch.float32, _SAMPLE_SHAPE)
        _assert_unit_interval(self, samples)


if __name__ == "__main__":
    unittest.main()
