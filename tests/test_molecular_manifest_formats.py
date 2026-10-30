"""Reference manifests accept both documented JSON layouts."""

import importlib
import json
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
            module = importlib.import_module("qc_mol_design.data.references")
            yield module
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)


def reference():
    return dict(reference_id="ref1", test_position=0, source_index=0,
                smiles="CC", canonical_smiles="CC", n_heavy_atoms=2,
                n_qubo_vars=1, formal_charge=0, size_stratum="small")


@pytest.mark.parametrize("wrapped", [True, False])
def test_loads_equivalent_manifest_layouts(loader, tmp_path, wrapped):
    rows = [reference()]
    path = tmp_path / "references.json"
    path.write_text(json.dumps({"references": rows} if wrapped else rows))
    result = loader.load_reference_manifest(path, expected_count=1)
    assert len(result) == 1
    assert result[0].reference_id == "ref1"
    assert result[0].canonical_smiles == "CC"


@pytest.mark.parametrize("payload", [None, 42, "bad", {}, {"references": None},
                                      {"references": {}}, [42]])
def test_rejects_invalid_manifest_layout(loader, tmp_path, payload):
    path = tmp_path / "references.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifest"):
        loader.load_reference_manifest(path, expected_count=1)


def test_bare_list_retains_count_and_duplicate_validation(loader, tmp_path):
    path = tmp_path / "references.json"
    path.write_text(json.dumps([reference(), reference()]))
    with pytest.raises(ValueError, match="expected"):
        loader.load_reference_manifest(path, expected_count=1)
    with pytest.raises(ValueError, match="duplicate"):
        loader.load_reference_manifest(path, expected_count=2)
