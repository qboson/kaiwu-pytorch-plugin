"""Reject unusable precomputed properties before molecular training."""

import importlib
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit", reason="Molecular CSV validation requires the optional RDKit dependency")


@pytest.fixture(params=["cim_sa", "sqa"])
def loader(request, monkeypatch):
    """Import the real loader from each independent backend."""
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


@pytest.mark.parametrize("property_name", ["qed", "logp", "sas"])
@pytest.mark.parametrize("bad_value", [np.inf, -np.inf, "not-a-number", np.nan])
def test_nonfinite_or_nonnumeric_properties_are_dropped(loader, tmp_path, property_name, bad_value):
    """A valid SMILES cannot make an invalid numeric property usable."""
    good = dict(smiles="CCO", qed=0.5, logp=1.2, sas=2.5)
    bad = dict(smiles="CC", qed=0.4, logp=1.0, sas=2.0)
    bad[property_name] = bad_value
    path = tmp_path / "molecules.csv"
    pd.DataFrame([good, bad]).to_csv(path, index=False)
    result = loader.load_zinc12k(str(path))
    assert result["smiles"].tolist() == ["CCO"]
    values = result[["qed", "logp", "sas"]].to_numpy(dtype=float)
    np.testing.assert_allclose(values, [[0.5, 1.2, 2.5]])
    assert np.isfinite(values).all()


def test_numeric_strings_and_valid_rows_remain_supported(loader, tmp_path):
    """Validation retains ordinary numeric CSVs, column normalization and SMILES filtering."""
    path = tmp_path / "molecules.csv"
    pd.DataFrame({
        " SMILES ": ["CCO", "CC", "invalid-smiles"],
        " QED ": ["0.5", "0.4", "0.3"],
        "logP": ["1.2", "-1.0", "0.0"],
        "SAS": ["2.5", "2.0", "1.0"],
    }).to_csv(path, index=False)
    result = loader.load_zinc12k(str(path))
    assert result["smiles"].tolist() == ["CCO", "CC"]
    np.testing.assert_allclose(result["logp"].to_numpy(dtype=float), [1.2, -1.0])
