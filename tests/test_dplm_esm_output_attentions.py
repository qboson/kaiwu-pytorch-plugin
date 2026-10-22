"""Regression test for the patched ESM attention and ``output_attentions``.

``_ModifiedEsmSelfAttention`` replaced the attention implementation with
``scaled_dot_product_attention`` and silently dropped the ``output_attentions``
flag (``del output_attentions``), returning only the context layer.  When a
caller asked for attentions, the stock ESM encoder indexed the missing entry:

    File ".../modeling_esm.py", line 633, in forward
        all_self_attentions = all_self_attentions + (layer_outputs[1],)
    IndexError: tuple index out of range

The patch must state its limitation explicitly instead of corrupting the
upstream contract.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402
from transformers.models.esm.configuration_esm import EsmConfig  # noqa: E402

from models.esm_patch import _ModifiedEsmSelfAttention  # noqa: E402


def _attention():
    config = EsmConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        position_embedding_type="absolute",
    )
    torch.manual_seed(0)
    return config, _ModifiedEsmSelfAttention(config).eval()


def test_output_attentions_is_rejected_explicitly():
    config, attention = _attention()
    hidden = torch.randn(1, 4, config.hidden_size)

    with pytest.raises(NotImplementedError):
        attention(hidden, output_attentions=True)


def test_default_call_returns_the_context_layer():
    config, attention = _attention()
    hidden = torch.randn(1, 4, config.hidden_size)

    outputs = attention(hidden)

    assert outputs[0].shape == hidden.shape
