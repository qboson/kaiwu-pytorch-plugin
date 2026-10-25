"""Generation-loop contract for the ESM2 evaluation helper."""

import os
import sys
import types

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPLM = os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm")
sys.path[:0] = [DPLM, os.path.join(DPLM, "workflows")]

# Other example tests bind their own top-level ``utils``/``models`` modules.
for _name in [m for m in list(sys.modules)
               if m in {"utils", "models"}
               or m.startswith("utils.")
               or m.startswith("models.")]:
    del sys.modules[_name]

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
import workflows.esm2_eval_helpers as helpers  # noqa: E402
if _added:
    sys.modules.pop("esm", None)


class StubTokenizer:
    """Records mask inputs and returns a fixed token row and decoded text."""

    def __init__(self):
        self.sequences = []

    def batch_encode_plus(self, sequences, **kwargs):
        self.sequences.extend(sequences)
        return {"input_ids": torch.tensor([[1, 3, 2, 4]])}

    def batch_decode(self, tokens, skip_special_tokens=True):
        return [" A C "]


class StubGenerator:
    """Records generation calls and exposes tokenizer/ids/device."""

    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.mask_id = 3
        self.device = torch.device("cpu")
        self.generate_calls = []

    def generate(self, tokens, max_steps, partial_masks):
        self.generate_calls.append((tuple(tokens.shape), max_steps, partial_masks.tolist()))
        return tokens


def test_each_record_is_masked_after_compaction_and_seeded_in_order(tmp_path, monkeypatch):
    """Sequences are compacted before masking and seeds advance per record."""
    seeds = []
    monkeypatch.setattr(helpers, "seed_torch", seeds.append)
    generator = StubGenerator()
    records = [("h1", "A C"), ("h2", "AC")]
    output = tmp_path / "generated.fasta"
    generated = helpers.run_generation_over_records(
        generator,
        records,
        max_steps=5,
        seed_base=100,
        output_fasta_path=output,
        label="guided",
    )

    assert generator.tokenizer.sequences == ["<mask><mask>", "<mask><mask>"]
    assert seeds == [101, 102]
    assert generator.generate_calls == [
        ((1, 4), 5, [[True, False, True, True]]),
        ((1, 4), 5, [[True, False, True, True]]),
    ]
    assert generated == [("h1", "AC"), ("h2", "AC")]
    assert output.read_text(encoding="utf-8").splitlines() == [
        ">seq_1 h1",
        "AC",
        ">seq_2 h2",
        "AC",
    ]
