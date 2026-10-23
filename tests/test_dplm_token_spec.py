"""Token-spec contract for the DPLM backbone wrapper."""

import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from models.backbone import build_dplm_token_spec  # noqa: E402


def test_spec_copies_every_special_id_and_the_tokenizer():
    """The generic spec mirrors the backbone's ids and tokenizer."""
    backbone = SimpleNamespace(
        mask_id=11,
        pad_id=10,
        bos_id=12,
        eos_id=13,
        x_id=14,
        tokenizer="TOKENIZER",
    )
    spec = build_dplm_token_spec(backbone)
    assert spec.mask_id == 11
    assert spec.pad_id == 10
    assert spec.bos_id == 12
    assert spec.eos_id == 13
    assert spec.x_id == 14
    assert spec.tokenizer == "TOKENIZER"
