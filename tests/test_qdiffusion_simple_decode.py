"""Decode contract for the minimal Q-Diffusion example."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIFF = os.path.join(REPO_ROOT, "example", "qdiffusion")
sys.path[:0] = [QDIFF, os.path.join(QDIFF, "dplm"), os.path.join(QDIFF, "simple")]

import torch  # noqa: E402

import simple_generate_example as example  # noqa: E402


class StubTokenizer:
    """Returns one canned decoded string."""

    def __init__(self, text):
        self.text = text
        self.calls = []

    def batch_decode(self, tokens, skip_special_tokens=True):
        self.calls.append((tokens, skip_special_tokens))
        return [self.text]


class StubGenerator:
    """Holds the tokenizer the decode helper reads."""

    def __init__(self, text):
        self.tokenizer = StubTokenizer(text)


def test_decoded_text_is_compacted_and_stripped():
    """Tokenizer spacing and surrounding whitespace are removed."""
    generator = StubGenerator("  A C G T \n")
    assert example.decode_tokens(generator, torch.zeros(1, 4)) == "ACGT"


def test_special_tokens_are_skipped_in_the_decode_call():
    """The helper asks the tokenizer to drop special tokens."""
    generator = StubGenerator("ACGT")
    example.decode_tokens(generator, torch.zeros(1, 4))
    tokens, skip = generator.tokenizer.calls[0]
    assert skip is True
    assert tuple(tokens.shape) == (1, 4)
