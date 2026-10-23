"""Encoding-path contract for the DPLM feature encoder."""

import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from models.common import DPLMFeatureEncoder  # noqa: E402


class FakeNet:
    """Records call kwargs and returns distinct hidden/logit payloads."""

    def __init__(self):
        self.calls = []
        self.hidden = torch.full((1, 2, 3), 7.0)
        self.logits = torch.full((1, 2, 5), 9.0)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"last_hidden_state": self.hidden, "logits": self.logits}


def build_encoder():
    """Wraps one fake net in the backbone interface the encoder expects."""
    net = FakeNet()
    backbone = SimpleNamespace(net=net, tokenizer="TOKENIZER")
    return DPLMFeatureEncoder(backbone), net


def test_encode_tokens_forwards_ids_and_mask_and_returns_hidden_states():
    """The plain encoding path passes the mask and returns last_hidden_state."""
    encoder, net = build_encoder()
    tokens = torch.tensor([[1, 2]])
    mask = torch.tensor([[1, 0]])
    result = encoder.encode_tokens(tokens, attention_mask=mask)
    assert net.calls == [{"input_ids": tokens, "attention_mask": mask}]
    assert torch.equal(result, net.hidden)


def test_encode_tokens_allows_a_missing_mask():
    """Without a mask the keyword is still forwarded as ``None``."""
    encoder, net = build_encoder()
    tokens = torch.tensor([[1, 2]])
    encoder.encode_tokens(tokens)
    assert net.calls == [{"input_ids": tokens, "attention_mask": None}]


def test_encode_conditioned_forwards_all_three_inputs():
    """Conditioned encoding passes ids, embeddings and mask together."""
    encoder, net = build_encoder()
    ids = torch.tensor([[1, 2]])
    embeds = torch.zeros(1, 2, 3)
    mask = torch.tensor([[1, 1]])
    result = encoder.encode_conditioned(
        input_ids=ids, inputs_embeds=embeds, attention_mask=mask
    )
    assert net.calls == [
        {"input_ids": ids, "inputs_embeds": embeds, "attention_mask": mask}
    ]
    assert torch.equal(result, net.hidden)
