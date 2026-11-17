"""Length-aware training batches use all sequences with less padding."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tqdm")
pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from example.qdiffusion.dplm.workflows.workflow_helpers import build_data_loader_from_records


class ToyTokenizer:
    def __call__(self, sequences, **kwargs):
        lengths = [len(sequence) + 2 for sequence in sequences]
        ids = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
        for row, length in enumerate(lengths):
            ids[row, :length] = 1
        return {"input_ids": ids}


def collect(loader):
    headers = []
    padding = 0
    for batch in loader:
        headers.extend(batch["headers"])
        padding += int((batch["targets"] == 0).sum())
    return headers, padding


def test_length_buckets_preserve_every_record_and_reduce_padding():
    generator = SimpleNamespace(tokenizer=ToyTokenizer(), device="cpu")
    lengths = [8, 100] * 9
    records = [(str(index), "A" * length) for index, length in enumerate(lengths)]
    kwargs = {"batch_size": 4, "shuffle": True}

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(27)
        ordinary = build_data_loader_from_records(generator, records, **kwargs)
        _, ordinary_padding = collect(ordinary)
        torch.default_generator.manual_seed(27)
        bucketed = build_data_loader_from_records(
            generator, records, length_bucket_batches=8, **kwargs
        )
        first_headers, first_padding = collect(bucketed)
        second_headers, second_padding = collect(bucketed)

    assert len(bucketed) == len(ordinary) == 5
    assert sorted(first_headers) == sorted(second_headers) == sorted(h for h, _ in records)
    assert first_headers != second_headers
    assert 0 < ordinary_padding
    assert max(first_padding, second_padding) < ordinary_padding / 2


def test_nonpositive_bucket_size_is_rejected():
    generator = SimpleNamespace(tokenizer=ToyTokenizer(), device="cpu")
    with pytest.raises(ValueError, match="length_bucket_batches must be positive"):
        build_data_loader_from_records(
            generator, [("r", "AC")], batch_size=2, shuffle=True, length_bucket_batches=0
        )
