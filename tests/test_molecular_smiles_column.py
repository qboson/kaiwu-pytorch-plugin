"""Read the intended molecular column in both solver variants."""

import importlib.util
from pathlib import Path
import warnings

import pytest

pytest.importorskip("pandas")
pytest.importorskip("rdkit")


@pytest.fixture(params=["sqa", "cim_sa"])
def molecular_loader(request, monkeypatch):
    root = Path.cwd() / "example" / "molecular_design" / request.param
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        f"{request.param}_data_loader", root / "qc_mol_design" / "data" / "loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module.load_zinc12k


def test_exact_smiles_column_takes_precedence_over_an_earlier_alias(molecular_loader, tmp_path):
    path = tmp_path / "molecules.csv"
    path.write_text("original_smiles,SMILES,qed,logp,sas\nN,CCO,0.4,1.0,2.0\n", encoding="utf-8")

    data = molecular_loader(str(path))

    assert data["smiles"].tolist() == ["CCO"]


def test_a_single_legacy_alias_is_supported(molecular_loader, tmp_path):
    path = tmp_path / "molecules.csv"
    path.write_text("canonical_smiles,qed,logp,sas\nCCO,0.4,1.0,2.0\n", encoding="utf-8")

    assert molecular_loader(str(path))["smiles"].tolist() == ["CCO"]


def test_metadata_is_never_treated_as_a_molecule_column(molecular_loader, tmp_path):
    path = tmp_path / "molecules.csv"
    path.write_text("id,qed,logp,sas\nCCO,0.4,1.0,2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unambiguous SMILES alias"):
        molecular_loader(str(path))
