"""Explicit molecular split counts must not be replaced by defaults."""

import importlib
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=["cim_sa", "sqa"])
def loader(request, monkeypatch):
    """Exercise each independently shipped backend without leaking imports."""
    prefix = "qc_mol_design"
    saved = {k: v for k, v in sys.modules.items() if k == prefix or k.startswith(prefix + ".")}
    for name in saved:
        sys.modules.pop(name)
    root = Path(__file__).resolve().parents[1] / "example" / "molecular_design"
    monkeypatch.syspath_prepend(str(root / request.param))
    try:
        with warnings.catch_warnings():
            module = importlib.import_module("qc_mol_design.data.loader")
            yield module
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)


@pytest.mark.parametrize("counts", [(6, 0, 2), (0, 3, 2), (5, 2, 0), (0, 0, 0)])
def test_zero_counts_are_honored(loader, counts):
    """Zero disables a split instead of requesting the configured default."""
    data = pd.DataFrame({"id": np.arange(10)})
    splits = loader.split_dataset(data, *counts, seed=7)
    assert tuple(len(frame) for frame in splits) == counts
    combined = pd.concat(splits)
    assert len(combined["id"].unique()) == sum(counts)


@pytest.mark.parametrize("field", ["n_train", "n_valid", "n_test"])
def test_negative_counts_are_rejected(loader, field):
    """Negative iloc boundaries must not create overlapping train/test sets."""
    counts = dict(n_train=2, n_valid=2, n_test=2)
    counts[field] = -1
    with pytest.raises(ValueError, match=field):
        loader.split_dataset(pd.DataFrame({"id": range(10)}), **counts)


def test_defaults_and_numpy_integer_counts(loader, monkeypatch):
    """Only None selects defaults; NumPy integer counts remain supported."""
    for key, value in [("n_train", 4), ("n_valid", 2), ("n_test", 1)]:
        monkeypatch.setattr(loader.CFG, key, value)
    data = pd.DataFrame({"id": np.arange(10)})
    first = loader.split_dataset(data, seed=8)
    second = loader.split_dataset(data, np.int64(4), np.int64(2), np.int64(1), seed=8)
    assert tuple(map(len, first)) == (4, 2, 1)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("field", ["n_train", "n_valid", "n_test"])
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_counts_must_be_integers(loader, field, value):
    """Do not silently truncate fractional counts or interpret booleans as sizes."""
    counts = dict(n_train=2, n_valid=2, n_test=2)
    counts[field] = value
    with pytest.raises(TypeError, match=field):
        loader.split_dataset(pd.DataFrame({"id": range(10)}), **counts)
