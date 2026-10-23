"""Feature-encoder delegation contract for the DPLM energy model."""

import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from models.common import DPLMFeatureEncoder  # noqa: E402


class FakeNet:
    """Exposes an embedding lookup and a config hidden size."""

    def __init__(self):
        self.config = SimpleNamespace(hidden_size=32)
        self.embedding_calls = []

    def get_input_embeddings(self):
        return self.embed


def build_encoder():
    """Wraps one fake backbone whose tokenizer is a sentinel object."""
    net = FakeNet()

    def embed(tokens):
        net.embedding_calls.append(tokens)
        return tokens + 100

    net.embed = embed
    backbone = SimpleNamespace(net=net, tokenizer="TOKENIZER")
    return DPLMFeatureEncoder(backbone), net


def test_hidden_size_comes_from_the_wrapped_net_config():
    """The projector width is read from the backbone's config."""
    encoder, _net = build_encoder()
    assert encoder.hidden_size == 32


def test_tokenizer_is_reached_through_the_backbone():
    """The tokenizer property delegates to the backbone, not the net."""
    encoder, _net = build_encoder()
    assert encoder.tokenizer == "TOKENIZER"


def test_embed_tokens_uses_the_input_embeddings_module():
    """Token embedding goes through ``get_input_embeddings``."""
    encoder, net = build_encoder()
    tokens = torch.tensor([[1, 2]])
    embedded = encoder.embed_tokens(tokens)
    assert net.embedding_calls == [tokens]
    assert embedded.tolist() == [[101, 102]]
