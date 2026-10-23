"""Encoding contract for the simplified QVAE-MNIST encoder."""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

from model.networks import SimpleEncoder  # noqa: E402


class StubSmoothing:
    """Returns its quantile input unchanged, so samples equal the sigmoid."""

    def icdf(self, rho, q):
        return q


def build_encoder():
    """Builds a one-layer sigmoid encoder with deterministic weights."""
    encoder = SimpleEncoder(
        smoothing_distribution=StubSmoothing(),
        node_sequence=[[3, 2]],
        activation_fct=torch.sigmoid,
    )
    with torch.no_grad():
        encoder._layers[0].weight.copy_(torch.arange(6, dtype=torch.float32).view(2, 3))
        encoder._layers[0].bias.zero_()
    return encoder


def test_encode_applies_the_activation_after_the_layer():
    """The encoded value is sigmoid(Wx + b)."""
    encoder = build_encoder()
    x = torch.tensor([[1.0, 0.0, -1.0]])
    expected = torch.sigmoid(x @ encoder._layers[0].weight.T)
    assert torch.allclose(encoder.encode(x), expected)


def test_hierarchical_posterior_keeps_four_levels():
    """One posterior/sample pair is produced per hierarchy level."""
    encoder = build_encoder()
    posterior, samples = encoder.hierarchical_posterior(torch.zeros(2, 3))
    assert len(posterior) == 4
    assert len(samples) == 4
    assert encoder.num_latent_hierarchy_levels == 4
    for entry in posterior:
        assert isinstance(entry, StubSmoothing)
    for sample in samples:
        assert sample.shape == (2, 2)
        assert torch.all((sample >= 0) & (sample <= 1))


def test_decode_is_not_supported_on_the_encoder():
    """The encoder rejects the decoder-side operation."""
    encoder = build_encoder()
    with pytest.raises(NotImplementedError):
        encoder.decode(torch.zeros(1, 2))
