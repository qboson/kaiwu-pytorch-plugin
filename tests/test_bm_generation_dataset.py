"""Dataset contract for the BM generation example loader."""

import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "bm_generation"))

from data_loader import CSVDataset  # noqa: E402


def test_rows_are_returned_as_float32_tensors():
    """Every row becomes one float32 tensor of the row width."""
    dataset = CSVDataset([[1, 2, 3], [4, 5, 6]])
    assert len(dataset) == 2
    item = dataset[0]
    assert item.dtype == torch.float32
    assert item.tolist() == [1.0, 2.0, 3.0]


def test_numpy_rows_are_converted_to_float32_tensors():
    """Numpy input keeps its values and is converted to float32."""
    data = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    item = CSVDataset(data)[1]
    assert item.dtype == torch.float32
    assert item.tolist() == [2.5, 3.5]


def test_items_are_copies_of_the_underlying_rows():
    """Mutating a returned row does not write back into the source data."""
    rows = [[1.0, 2.0]]
    item = CSVDataset(rows)[0]
    item[0] = 99.0
    assert rows == [[1.0, 2.0]]


def test_negative_index_and_out_of_range():
    """Indexing follows normal sequence semantics."""
    dataset = CSVDataset([[1.0], [2.0], [3.0]])
    assert dataset[-1].tolist() == [3.0]
    with pytest.raises(IndexError):
        dataset[3]


def test_data_loader_stacks_rows_into_one_batch():
    """The default collate builds a (batch, features) float32 batch."""
    loader = DataLoader(CSVDataset([[1, 2], [3, 4], [5, 6]]), batch_size=3)
    batch = next(iter(loader))
    assert batch.shape == (3, 2)
    assert batch.dtype == torch.float32
