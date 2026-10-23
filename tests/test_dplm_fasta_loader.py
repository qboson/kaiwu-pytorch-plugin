"""Wiring contract for the DPLM record dataset and its data loader."""

import os
import sys

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

pytest.importorskip("tqdm")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from workflows.workflow_helpers import (  # noqa: E402
    FastaSequenceDataset,
    build_data_loader_from_records,
)


RECORDS = [("h1", "AAA"), ("h2", "BBBB"), ("h3", "CC")]


class StubTokenizer:
    """Deterministic tokenizer that records the arguments it receives."""

    def __init__(self):
        self.calls = []

    def __call__(self, sequences, **kwargs):
        self.calls.append((list(sequences), kwargs))
        width = max(len(sequence) for sequence in sequences) + 1
        rows = []
        for sequence in sequences:
            padding = [2] * (width - len(sequence) - 1)
            rows.append([1] + [3] * len(sequence) + padding)
        return {"input_ids": torch.tensor(rows)}


class StubGenerator:
    """Minimal generator exposing only what the collate function reads."""

    def __init__(self, device="cpu"):
        self.tokenizer = StubTokenizer()
        self.device = torch.device(device)


def test_dataset_exposes_records_by_index():
    """The dataset returns one header/sequence pair per index."""
    dataset = FastaSequenceDataset(RECORDS)
    assert len(dataset) == 3
    assert dataset[0] == {"header": "h1", "sequence": "AAA"}
    assert dataset[-1] == {"header": "h3", "sequence": "CC"}


def test_loader_batches_headers_and_moves_tokens_to_the_generator_device():
    """Collate keeps header order and returns padded targets on the device."""
    generator = StubGenerator()
    loader = build_data_loader_from_records(
        generator, RECORDS, batch_size=2, shuffle=False
    )
    batches = list(loader)
    assert [batch["headers"] for batch in batches] == [["h1", "h2"], ["h3"]]
    assert batches[0]["targets"].shape == (2, 5)
    assert batches[1]["targets"].shape == (1, 3)
    sequences, kwargs = generator.tokenizer.calls[0]
    assert sequences == ["AAA", "BBBB"]
    assert kwargs["add_special_tokens"] is True
    assert kwargs["padding"] is True


def test_loader_moves_targets_to_a_non_cpu_device():
    """Targets follow ``generator.device``, not the tokenizer output device."""
    generator = StubGenerator(device="meta")
    loader = build_data_loader_from_records(
        generator, RECORDS, batch_size=3, shuffle=False
    )
    batch = next(iter(loader))
    assert batch["targets"].device.type == "meta"


def test_shuffle_flag_selects_the_sampler():
    """``shuffle`` is wired to the sampler, not only to the batch sampler."""
    generator = StubGenerator()
    shuffled = build_data_loader_from_records(
        generator, RECORDS, batch_size=1, shuffle=True
    )
    ordered = build_data_loader_from_records(
        generator, RECORDS, batch_size=1, shuffle=False
    )
    assert isinstance(shuffled.sampler, RandomSampler)
    assert isinstance(ordered.sampler, SequentialSampler)
