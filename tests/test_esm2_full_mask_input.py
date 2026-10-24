"""Mask-input construction contract for the ESM2 distance workflow."""

import os
import sys
import types

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPLM = os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm")
sys.path[:0] = [DPLM, os.path.join(DPLM, "workflows")]

# Other example tests bind their own top-level ``models``/``utils``/``workflows``;
# the DPLM fallback imports need this example's packages to win the lookup.
for _name in [m for m in list(sys.modules)
               if m in {"models", "utils", "workflows"}
               or m.startswith("models.")
               or m.startswith("utils.")
               or m.startswith("workflows.")]:
    del sys.modules[_name]

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
from workflows.esm2_eval_helpers import build_full_mask_input  # noqa: E402
if _added:
    sys.modules.pop("esm", None)


class StubTokenizer:
    """Records ``batch_encode_plus`` calls and returns a fixed token row."""

    def __init__(self):
        self.calls = []

    def batch_encode_plus(self, sequences, **kwargs):
        self.calls.append((list(sequences), kwargs))
        return {"input_ids": torch.tensor([[9, 1, 2, 3, 9]])}


class StubGenerator:
    """Exposes the tokenizer and device the helper reads."""

    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.device = torch.device("cpu")


def test_mask_text_length_matches_the_sequence_length():
    """The helper asks the tokenizer for one ``<mask>`` per position."""
    generator = StubGenerator()
    build_full_mask_input(generator, 3)
    sequences, kwargs = generator.tokenizer.calls[0]
    assert sequences == ["<mask><mask><mask>"]
    assert kwargs["add_special_tokens"] is True
    assert kwargs["padding"] == "longest"
    assert kwargs["return_tensors"] == "pt"


def test_encoded_ids_are_returned_for_the_requested_length():
    """The encoded ids are passed through unchanged."""
    generator = StubGenerator()
    tokens = build_full_mask_input(generator, 3)
    assert tokens.tolist() == [[9, 1, 2, 3, 9]]
