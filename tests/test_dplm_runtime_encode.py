"""Tokenization contract for the DPLM runtime helpers."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.runtime import encode_sequence  # noqa: E402


class StubTokenizer:
    """Records the text it is asked to tokenize."""

    def __init__(self):
        self.seen = []

    def __call__(self, sequence, **kwargs):
        self.seen.append((sequence, kwargs))
        return {"input_ids": torch.tensor([[1] + [3] * len(sequence)])}


class StubGenerator:
    """Exposes only the tokenizer/device attributes the helper reads."""

    def __init__(self, device="cpu"):
        self.tokenizer = StubTokenizer()
        self.device = torch.device(device)


def test_max_length_truncates_the_text_before_tokenization():
    """Truncation is applied to residues, not to token ids."""
    generator = StubGenerator()
    encode_sequence(generator, "ABCDE", max_length=3)
    sequence, kwargs = generator.tokenizer.seen[0]
    assert sequence == "ABC"
    assert kwargs["add_special_tokens"] is True


def test_without_max_length_the_whole_sequence_is_encoded():
    """``None`` means no truncation."""
    generator = StubGenerator()
    tokens = encode_sequence(generator, "ABCDE")
    assert generator.tokenizer.seen[0][0] == "ABCDE"
    assert tokens.shape == (1, 6)


def test_tokens_follow_the_generator_device():
    """The encoded batch is moved to ``generator.device``."""
    generator = StubGenerator(device="meta")
    tokens = encode_sequence(generator, "ABC", max_length=10)
    assert tokens.device.type == "meta"
