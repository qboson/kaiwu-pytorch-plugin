"""Nonfinite external condition values must not silently become one-hot labels."""

import importlib
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest


@pytest.fixture(params=["sqa", "cim_sa"])
def binning(request, monkeypatch):
    pytest.importorskip("pandas", exc_type=ImportError)
    base = Path(__file__).resolve().parents[1] / "example" / "molecular_design"
    monkeypatch.syspath_prepend(str(base / request.param))
    for name in tuple(sys.modules):
        if name == "qc_mol_design" or name.startswith("qc_mol_design."):
            monkeypatch.delitem(sys.modules, name)
    with warnings.catch_warnings():
        module = importlib.import_module("qc_mol_design.data.loader")
    yield module
    for name in tuple(sys.modules):
        if name == "qc_mol_design" or name.startswith("qc_mol_design."):
            sys.modules.pop(name)


@pytest.mark.parametrize("property_index, prop", [(0, "qed"), (1, "logp"), (2, "sas")])
def test_invalid_conditions_never_produce_a_label_vector(binning, property_index, prop):
    for invalid in (np.nan, np.inf, -np.inf):
        values = [0.5, 0.0, 2.0]
        values[property_index] = invalid
        with pytest.raises(ValueError, match=rf"{prop} values must be finite"):
            binning.make_label_vector(*values)


def test_finite_outliers_keep_edge_bins_and_existing_one_hot_layout(binning):
    for prop in ("qed", "logp", "sas"):
        edges = binning.PROP_BINS[prop]
        actual = binning.discretize([edges[0] - 10, edges[1], edges[-1] + 10], prop)
        np.testing.assert_array_equal(actual, [0, 1, binning.N_BINS_PER_PROP[prop] - 1])
    label = binning.make_label_vector(0.5, -1.0, 2.0)
    assert label.sum() == 3
    assert len(label) == sum(binning.N_BINS_PER_PROP.values())
