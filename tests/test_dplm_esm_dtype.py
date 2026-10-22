"""Regression tests for the patched ESM attention dtype handling."""

import os
import sys

import pytest
import torch

pytest.importorskip("transformers")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example.qdiffusion.dplm.models.esm_patch import (  # noqa: E402  pylint: disable=wrong-import-position
    _EsmForDPLM,
)


def _tiny_rotary_model(monkeypatch):
    from transformers import EsmConfig

    class _DummyTokenizer:
        mask_token_id = 32
        pad_token_id = 1
        cls_token_id = 0
        bos_token_id = 0
        eos_token_id = 2
        _token_to_id = {"X": 24}

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )
    config = EsmConfig(
        vocab_size=33,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        position_embedding_type="rotary",
        token_dropout=False,
        pad_token_id=1,
    )
    model = _EsmForDPLM(config, dropout=0.0)
    model.eval()
    return model


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_after_dtype_cast(dtype, monkeypatch):
    """Casting the model after a float32 forward must not break attention."""
    model = _tiny_rotary_model(monkeypatch)
    input_ids = torch.randint(4, 30, (1, 10))

    with torch.no_grad():
        model(input_ids)  # populates the rotary cos/sin cache in float32
        model = model.to(dtype)
        outputs = model(input_ids)

    assert outputs["logits"].dtype == dtype
    assert torch.isfinite(outputs["logits"].float()).all()


def test_mixed_dtype_does_not_change_float32_results(monkeypatch):
    model = _tiny_rotary_model(monkeypatch)
    input_ids = torch.randint(4, 30, (1, 10))

    with torch.no_grad():
        first = model(input_ids)["logits"]
        second = model(input_ids)["logits"]

    assert torch.equal(first, second)
