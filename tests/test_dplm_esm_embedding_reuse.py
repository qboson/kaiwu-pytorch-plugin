"""Repeated protein sequences need one ESM2 forward regardless of headers."""

import pytest
import torch
from torch import nn

esm = pytest.importorskip("esm")
pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from example.qdiffusion.dplm.workflows.esm2_eval_helpers import embed_sequences


class CountingEncoder(nn.Module):
    num_layers = 1

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, tokens, *, repr_layers, return_contacts):
        self.batch_sizes.append(tokens.size(0))
        return {"representations": {1: tokens.to(torch.float32).unsqueeze(-1)}}


def test_duplicates_are_embedded_once_and_keep_every_header():
    alphabet = esm.Alphabet.from_architecture("ESM-1b")
    model = CountingEncoder().eval()
    records = [("a", "AC"), ("b", "A C"), ("c", "GGG")]

    result = embed_sequences(records, model=model, alphabet=alphabet, device=torch.device("cpu"),
                             batch_size=1, pooling="mean")

    assert list(result) == ["a", "b", "c"]
    assert model.batch_sizes == [1, 1]
    expected = (alphabet.get_idx("A") + alphabet.get_idx("C")) / 2
    torch.testing.assert_close(result["a"], torch.tensor([expected], dtype=torch.float32))
    torch.testing.assert_close(result["b"], result["a"])


def test_reference_and_generated_sets_share_cached_representations():
    alphabet = esm.Alphabet.from_architecture("ESM-1b")
    model = CountingEncoder().eval()
    cache = {}
    kwargs = dict(model=model, alphabet=alphabet, device=torch.device("cpu"),
                  batch_size=2, pooling="mean", sequence_cache=cache)
    reference = embed_sequences([("ref", "AC")], **kwargs)
    candidates = embed_sequences([("same sequence", "AC"), ("different", "GGG")], **kwargs)

    assert model.batch_sizes == [1, 1]
    torch.testing.assert_close(candidates["same sequence"], reference["ref"])
    assert not torch.equal(candidates["different"], reference["ref"])
