"""Embedding-pooling contract for the ESM2 distance workflow."""

import os
import sys
import types

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
    """Returns a fixed token matrix and reports padding id 0."""

    padding_idx = 0

    def get_batch_converter(self):
        def convert(batch):
            headers = [header for header, _sequence in batch]
            sequences = [sequence for _, sequence in batch]
            tokens = torch.tensor([[9, 1, 2, 3, 9], [9, 4, 5, 0, 0]])
            return headers, sequences, tokens

        return convert


class StubModel(torch.nn.Module):
    """Exposes ``num_layers`` and returns per-token representations."""

    num_layers = 2

    def forward(self, tokens, repr_layers, return_contacts):
        assert repr_layers == [2]
        assert return_contacts is False
        representations = tokens.float().unsqueeze(-1).repeat(1, 1, 2)
        return {"representations": {2: representations}}


RECORDS = [("h1", "ACG"), ("h2", "AC")]


def test_mean_pooling_averages_the_residue_positions():
    """Residues exclude the BOS position and stop before padding."""
    embeddings = embed_sequences(
        RECORDS,
        model=StubModel(),
        alphabet=StubAlphabet(),
        device=torch.device("cpu"),
        batch_size=2,
        pooling="mean",
    )
    assert set(embeddings) == {"h1", "h2"}
    assert embeddings["h1"].tolist() == [2.0, 2.0]
    assert embeddings["h2"].tolist() == [4.5, 4.5]


def test_cls_and_eos_modes_pick_the_boundary_positions():
    """CLS takes position 0; EOS takes the last non-padding position."""
    cls_embeddings = embed_sequences(
        RECORDS,
        model=StubModel(),
        alphabet=StubAlphabet(),
        device=torch.device("cpu"),
        batch_size=2,
        pooling="cls",
    )
    assert cls_embeddings["h1"].tolist() == [9.0, 9.0]
    eos_embeddings = embed_sequences(
        RECORDS,
        model=StubModel(),
        alphabet=StubAlphabet(),
        device=torch.device("cpu"),
        batch_size=2,
        pooling="eos",
    )
    assert eos_embeddings["h1"].tolist() == [9.0, 9.0]
    assert eos_embeddings["h2"].tolist() == [5.0, 5.0]
