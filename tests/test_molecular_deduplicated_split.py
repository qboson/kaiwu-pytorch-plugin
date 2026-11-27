"""Keep alternate SMILES for a molecule out of disjoint benchmark splits."""

import importlib
from pathlib import Path
import sys
import warnings

import pytest


@pytest.fixture(params=["sqa", "cim_sa"])
def loader(request, monkeypatch):
    pytest.importorskip("pandas", exc_type=ImportError)
    pytest.importorskip("rdkit.Chem", exc_type=ImportError)
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


def test_canonical_duplicate_rows_are_removed_without_merging_stereoisomers(loader):
    import pandas as pd
    from rdkit import Chem

    data = pd.DataFrame({
        "smiles": ["CCO", "OCC", "F[C@H](Cl)Br", "F[C@@H](Cl)Br", "CCC"],
        "record": [1, 2, 3, 4, 5],
    })
    splits = loader.split_dataset(data, 2, 1, 1, seed=9, deduplicate_smiles=True)
    repeat = loader.split_dataset(data, 2, 1, 1, seed=9, deduplicate_smiles=True)
    for actual, expected in zip(splits, repeat):
        pd.testing.assert_frame_equal(actual, expected)
    merged = pd.concat(splits)
    assert set(merged["record"]) == {1, 3, 4, 5}
    canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in merged["smiles"]]
    assert len(canonical) == len(set(canonical))
    assert len(data) == 5


def test_capacity_is_checked_after_deduplication_and_legacy_split_is_available(loader):
    import pandas as pd

    data = pd.DataFrame({"smiles": ["CCO", "OCC", "CCC"]})
    with pytest.raises(ValueError, match="has 2 molecules but 3 requested"):
        loader.split_dataset(data, 1, 1, 1, deduplicate_smiles=True)
    assert tuple(map(len, loader.split_dataset(data, 1, 1, 1))) == (1, 1, 1)
