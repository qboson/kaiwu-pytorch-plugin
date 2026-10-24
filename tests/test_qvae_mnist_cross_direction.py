"""Cross-direction rejection contract for the QVAE-MNIST network classes."""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

from model.networks import BasicDecoder, BasicEncoder  # noqa: E402


def test_encoder_rejects_decoding():
    """The encoder refuses the decoder-side operation."""
    encoder = BasicEncoder(node_sequence=[[2, 2]], activation_fct=None)
    with pytest.raises(NotImplementedError, match="Decoder not implemented for encoder"):
        encoder.decode(torch.zeros(1, 2))


def test_decoder_rejects_encoding():
    """The decoder refuses the encoder-side operation."""
    decoder = BasicDecoder(node_sequence=[[2, 2]], activation_fct=None)
    with pytest.raises(NotImplementedError, match="Encoder not implemented for decoder"):
        decoder.encode(torch.zeros(1, 2))
