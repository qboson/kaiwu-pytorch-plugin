"""Reject reference metadata that would falsify strict-run provenance."""
import json
import os
from pathlib import Path
import sys

import pandas as pd
import pytest
from rdkit import Chem

SOURCE = os.environ.get("QC_MODEL_SOURCE", "sqa")
if SOURCE not in ("sqa", "cim_sa"):
    raise ValueError("QC_MODEL_SOURCE must be sqa or cim_sa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / SOURCE))
from qc_mol_design.data.references import load_reference_manifest


@pytest.fixture
def reference():
    mol = Chem.MolFromSmiles("C[NH3+]")
    return dict(reference_id="ref001", test_position=1, source_index=42,
                smiles="C[NH3+]", canonical_smiles=Chem.MolToSmiles(mol),
                n_heavy_atoms=2, n_qubo_vars=1, formal_charge=1, size_stratum="small")


def load(tmp_path, reference, test_df=None):
    path = tmp_path / "references.json"
    path.write_text(json.dumps({"references": [reference]}), encoding="utf-8")
    return load_reference_manifest(path, test_df=test_df, expected_count=1)


def test_valid_metadata_uses_position_not_dataframe_index(tmp_path, reference):
    frame = pd.DataFrame({"smiles": ["CCO", "[NH3+]C"]}, index=[10, 20])
    assert load(tmp_path, reference, frame)[0].test_position == 1


def test_charge_mismatch_is_rejected_without_test_split(tmp_path, reference):
    reference["formal_charge"] = 0
    with pytest.raises(ValueError, match="formal.charge"):
        load(tmp_path, reference)


@pytest.mark.parametrize("position", [-1, 0, 2, 1.0, True])
def test_wrong_or_invalid_test_position_is_rejected(tmp_path, reference, position):
    reference["test_position"] = position
    frame = pd.DataFrame({"smiles": ["CCO", "C[NH3+]"]}, index=[10, 20])
    with pytest.raises(ValueError, match="test.position"):
        load(tmp_path, reference, frame)


def test_position_cannot_be_verified_without_test_split(tmp_path, reference):
    # A standalone manifest remains usable; only molecular invariants are checked.
    reference["test_position"] = 100
    assert len(load(tmp_path, reference)) == 1


def test_absent_reference_retains_clear_error(tmp_path, reference):
    with pytest.raises(ValueError, match="absent from fixed test split"):
        load(tmp_path, reference, pd.DataFrame({"smiles": ["CCO", "CC"]}))
