"""Empty computed-property inputs retain the molecular table schema."""
import importlib
from pathlib import Path
import sys
import warnings

import pytest

pytest.importorskip("rdkit", reason="Reference validation requires the optional RDKit dependency")


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




@pytest.mark.parametrize("kind", ["header_only", "no_training_rows", "precomputed"])
def test_empty_csv_retains_schema(loader, tmp_path, kind):
    import pandas as pd
    path = tmp_path / "empty.csv"
    if kind == "header_only":
        pd.DataFrame(columns=["smiles"]).to_csv(path, index=False)
    elif kind == "no_training_rows":
        pd.DataFrame({"type": ["test"], "smiles": ["CC"]}).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=["smiles", "qed", "logp", "sas"]).to_csv(path, index=False)
    data = loader.load_zinc12k(path)
    assert data.empty
    assert list(data.columns) == ["smiles", "qed", "logp", "sas"]


def test_nonempty_property_computation_is_retained(loader, tmp_path):
    import pandas as pd
    import numpy as np
    path = tmp_path / "molecules.csv"
    pd.DataFrame({"smiles": ["CC", "not_a_smiles"]}).to_csv(path, index=False)
    data = loader.load_zinc12k(path)
    assert data["smiles"].tolist() == ["CC"]
    assert np.isfinite(data[["qed", "logp", "sas"]].to_numpy()).all()
