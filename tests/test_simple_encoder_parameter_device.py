"""Regression tests for ``SimpleEncoder.hierarchical_posterior`` device alignment.

``hierarchical_posterior`` encodes ``x`` with the encoder's linear weights and
then draws ``rho = torch.rand(q.size())``. That factory always uses the
default CPU device, so ``rho`` is mixed with the encoded activation inside
``smoothing_distribution.icdf`` whenever the weights live somewhere else.

A parent ``nn.Module._apply`` (the path ``nn.Module.to`` uses for children;
merged #164) moves those weights and does not call ``SimpleEncoder.to``. The
same mix happens when ``x`` is already on the parameter device: the bug is the
sample, not only the caller batch.

Out of scope: library ``QVAE.forward`` (#1370), cell ``_encode`` (#1371), and
the MNIST trainer / feature-extractor placement fixes.
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
    """Inverse CDF that actually consumes both ``rho`` and ``q``.

    The result stays inside ``[0, 1]`` when both inputs do, which is the
    range a smoothing quantile is supposed to produce.
    """

    def icdf(self, rho, q):
        """Mix the uniform sample with the encoded Bernoulli probability."""
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


def _move_parameters_to_meta(module):
    """Move parameters the way a parent does: ``_apply``, not ``.to``."""
    module._apply(lambda tensor: tensor.to(torch.device("meta")))


class TestSimpleEncoderPosteriorParameterDevice(unittest.TestCase):
    """Mechanism tests that run on every machine, including CPU-only CI."""

    def test_cpu_posterior_stays_in_unit_interval(self):
        """An unmoved CPU encoder still returns one finite sample per level."""
        encoder = _encoder()
        features = torch.zeros(2, 3)

        _posterior, samples = encoder.hierarchical_posterior(features)

        self.assertEqual(len(samples), encoder.num_latent_hierarchy_levels)
        self.assertEqual(samples[0].shape, (2, 2))
        self.assertEqual(samples[0].device, torch.device("cpu"))
        self.assertEqual(samples[0].dtype, torch.float32)
        for sample in samples:
            self.assertTrue(torch.isfinite(sample).all().item())
            self.assertTrue(torch.all((sample >= 0.0) & (sample <= 1.0)).item())

    def test_parent_apply_accepts_cpu_batch(self):
        """CPU features must be scored once the linear weights live on meta."""
        encoder = _encoder()
        parent = nn.Module()
        parent.encoder = encoder
        _move_parameters_to_meta(parent)
        features = torch.ones(2, 3)

        _posterior, samples = encoder.hierarchical_posterior(features)

        self.assertEqual(features.device, torch.device("cpu"))
        self.assertTrue(torch.equal(features, torch.ones(2, 3)))
        self.assertEqual(samples[0].device, torch.device("meta"))
        self.assertEqual(samples[0].shape, (2, 2))
        self.assertEqual(samples[0].dtype, torch.float32)
        self.assertEqual(
            next(encoder.parameters()).device,
            torch.device("meta"),
        )

    def test_rho_follows_parameters_when_input_is_already_there(self):
        """``rho`` must not stay on CPU after the activation moved with the weights."""
        encoder = _encoder()
        _move_parameters_to_meta(encoder)
        features = torch.ones(2, 3, device=torch.device("meta"))

        _posterior, samples = encoder.hierarchical_posterior(features)

        self.assertEqual(features.device, torch.device("meta"))
        self.assertEqual(samples[0].device, torch.device("meta"))
        self.assertEqual(samples[0].dtype, torch.float32)
        self.assertEqual(len(samples), 4)

    def test_parent_apply_skips_encoder_to(self):
        """Parent ``_apply`` must not be required to call ``SimpleEncoder.to``."""
        encoder = _encoder(_RecordingEncoder)
        parent = nn.Module()
        parent.encoder = encoder
        _move_parameters_to_meta(parent)

        _posterior, samples = encoder.hierarchical_posterior(torch.ones(2, 3))

        self.assertEqual(encoder.direct_to_calls, [])
        self.assertEqual(samples[0].device, torch.device("meta"))

    def test_parameter_free_encoder_follows_the_input_device(self):
        """With no weights, the uniform sample follows the caller activation."""
        encoder = _parameter_free_encoder()
        features = torch.zeros(2, 3, device=torch.device("meta"))

        _posterior, samples = encoder.hierarchical_posterior(features)

        self.assertIsNone(next(encoder.parameters(), None))
        self.assertEqual(samples[0].device, torch.device("meta"))
        self.assertEqual(samples[0].shape, features.shape)

    def test_double_parameters_keep_feature_dtype(self):
        """Moving float64 weights must not recast the caller features or ``rho``."""
        encoder = _encoder(dtype=torch.float64)
        _move_parameters_to_meta(encoder)
        features = torch.ones(2, 3, dtype=torch.float64)

        _posterior, samples = encoder.hierarchical_posterior(features)

        self.assertEqual(features.dtype, torch.float64)
        self.assertEqual(samples[0].dtype, torch.float64)
        self.assertEqual(samples[0].device, torch.device("meta"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSimpleEncoderPosteriorCudaParameterDevice(unittest.TestCase):
    """End-to-end coverage of a parent move onto CUDA."""

    def test_parent_move_accepts_cpu_batch(self):
        """CPU features must yield CUDA samples after a parent ``_apply``."""
        encoder = _encoder()
        parent = nn.Module()
        parent.encoder = encoder
        parent._apply(lambda tensor: tensor.to(torch.device("cuda")))
        features = torch.ones(2, 3)

        _posterior, samples = encoder.hierarchical_posterior(features)

        self.assertEqual(features.device, torch.device("cpu"))
        self.assertEqual(samples[0].device.type, "cuda")
        self.assertEqual(samples[0].shape, (2, 2))
        self.assertTrue(torch.isfinite(samples[0]).all().item())


if __name__ == "__main__":
    unittest.main()
