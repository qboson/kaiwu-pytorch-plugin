"""Embedding-input validation contract for the ESM2 distance workflow."""

import os
import sys
import types

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPLM = os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm")
sys.path[:0] = [DPLM, os.path.join(DPLM, "workflows")]

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
from workflows.esm2_eval_helpers import embed_sequences  # noqa: E402
if _added:
    sys.modules.pop("esm", None)


class StubAlphabet:
    """Returns a caller-provided token matrix."""

    padding_idx = 0

    def __init__(self, tokens):
        self.tokens = tokens

    def get_batch_converter(self):
        def convert(batch):
            headers = [header for header, _sequence in batch]
            sequences = [sequence for _, sequence in batch]
            return headers, sequences, self.tokens

        return convert


class StubModel(torch.nn.Module):
    """Returns one representation per token."""

    num_layers = 1

    def forward(self, tokens, repr_layers, return_contacts):
        return {"representations": {1: tokens.float().unsqueeze(-1).repeat(1, 1, 2)}}


def run(records, tokens, pooling="mean"):
    """Calls the helper with the stub stack."""
    return embed_sequences(
        records,
        model=StubModel(),
        alphabet=StubAlphabet(tokens),
        device=torch.device("cpu"),
        batch_size=2,
        pooling=pooling,
    )


def test_empty_record_set_returns_an_empty_mapping():
    """No records means no work and an empty result."""
    assert run([], torch.zeros(0, 0, dtype=torch.long)) == {}


def test_unsupported_pooling_mode_is_rejected():
    """Only mean/cls/bos/eos are accepted."""
    with pytest.raises(ValueError, match="Unsupported pooling mode"):
        run([("h", "AC")], torch.tensor([[9, 1, 2, 9]]), pooling="max")


def test_empty_sequence_names_the_offending_header():
    """An empty residue string fails with the record header in the message."""
    with pytest.raises(ValueError, match="empty sequence for header 'bad'"):
        run([("bad", "")], torch.tensor([[9, 9]]))
