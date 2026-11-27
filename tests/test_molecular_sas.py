"""SAS failures must not fabricate training labels or generated target hits."""

import importlib
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
Chem = pytest.importorskip("rdkit.Chem")
sascorer = pytest.importorskip("rdkit.Contrib.SA_Score.sascorer")
pytest.importorskip("sklearn")
pytest.importorskip("tqdm")

BIPHENYL = "c1ccc(-c2ccccc2)cc1"


@pytest.fixture(params=["cim_sa", "sqa"])
def molecular_modules(request, monkeypatch):
    """Import each real backend without mixing their identically named packages."""
    root = Path(__file__).resolve().parents[1] / "example/molecular_design"
    saved = {name: module for name, module in sys.modules.items()
             if name == "qc_mol_design" or name.startswith("qc_mol_design.")}
    for name in saved:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(root / request.param))
    try:
        loader = importlib.import_module("qc_mol_design.data.loader")
        designer = importlib.import_module("qc_mol_design.optimization.designer")
        yield loader, designer
    finally:
        for name in list(sys.modules):
            if name == "qc_mol_design" or name.startswith("qc_mol_design."):
                del sys.modules[name]
        sys.modules.update(saved)


def test_real_sas_score(molecular_modules):
    """Biphenyl's true SAS is 1.0; the removed heavy-atom proxy gives 2.2."""
    from rdkit.Chem import Crippen, QED

    loader, designer = molecular_modules
    mol = Chem.MolFromSmiles(BIPHENYL)
    expected = {"qed": QED.qed(mol), "logp": Crippen.MolLogP(mol), "sas": 1.0}
    assert loader._compute_properties(pd.Series([BIPHENYL])).iloc[0].to_dict() == pytest.approx(expected)
    assert designer.compute_rdkit_props(BIPHENYL) == pytest.approx(expected)


@pytest.mark.parametrize("entry_point", ["loader", "designer"])
def test_missing_sas_scorer_is_diagnosed(molecular_modules, monkeypatch, entry_point):
    loader, designer = molecular_modules
    monkeypatch.setitem(sys.modules, "rdkit.Contrib.SA_Score", None)
    with pytest.raises(ImportError, match="SA_Score"):
        if entry_point == "loader":
            loader._compute_properties(pd.Series([BIPHENYL]))
        else:
            designer.compute_rdkit_props(BIPHENYL)


def test_failed_sas_score_is_missing(molecular_modules, monkeypatch):
    loader, designer = molecular_modules
    monkeypatch.setattr(sascorer, "calculateScore", Mock(side_effect=RuntimeError("bad molecule")))
    assert loader._compute_properties(pd.Series([BIPHENYL])).iloc[0].isna().all()
    assert designer.compute_rdkit_props(BIPHENYL) is None


def test_precomputed_csv_does_not_need_sas_scorer(molecular_modules, monkeypatch, tmp_path):
    loader, _ = molecular_modules
    monkeypatch.setitem(sys.modules, "rdkit.Contrib.SA_Score", None)
    path = tmp_path / "precomputed.csv"
    expected = pd.DataFrame({"smiles": [BIPHENYL], "qed": [0.5], "logp": [4.0], "sas": [1.0]})
    expected.to_csv(path, index=False)
    pd.testing.assert_frame_equal(loader.load_zinc12k(str(path)), expected)


def test_failed_sas_rows_cannot_become_training_labels(molecular_modules, monkeypatch, tmp_path):
    loader, _ = molecular_modules
    original = sascorer.calculateScore

    def score(mol):
        if mol.GetNumHeavyAtoms() == 12:
            raise RuntimeError("bad molecule")
        return original(mol)

    monkeypatch.setattr(sascorer, "calculateScore", score)
    path = tmp_path / "smiles.csv"
    pd.DataFrame({"smiles": [BIPHENYL, "CCO"]}).to_csv(path, index=False)
    data = loader.load_zinc12k(str(path))
    assert data["smiles"].tolist() == ["CCO"]
    assert data["sas"].iloc[0] == pytest.approx(original(Chem.MolFromSmiles("CCO")))


@pytest.mark.parametrize("score_fails", [False, True])
def test_design_never_counts_proxy_sas_as_target_hit(molecular_modules, monkeypatch, score_fails):
    _, designer = molecular_modules
    if score_fails:
        monkeypatch.setattr(sascorer, "calculateScore", Mock(side_effect=RuntimeError("bad molecule")))
    monkeypatch.setattr(designer, "adj_to_smiles", lambda *args, **kwargs: BIPHENYL)
    monkeypatch.setattr(designer.CFG, "lambda_mode", "fixed")
    monkeypatch.setenv("QC_QUBO_SOLVER", "neal")
    monkeypatch.setenv("QC_CANDIDATE_SELECTION", "valid_energy")
    monkeypatch.delenv("QC_HISTORY_PREFIX_FILE", raising=False)
    mol = Chem.MolFromSmiles(BIPHENYL)
    bits = "".join(str(int(mol.GetBondBetweenAtoms(i, j) is not None))
                   for i in range(mol.GetNumAtoms()) for j in range(i + 1, mol.GetNumAtoms()))
    runner = designer.MolecularDesigner.__new__(designer.MolecularDesigner)
    runner._fingerprint = Mock(return_value=np.zeros(1))
    runner._free_energy = Mock(return_value=-1.0)

    result = runner.design(
        BIPHENYL, target_prop="sas", target_range=(2.122, 2.316),
        qubo_mode="strict", n_explore=1, n_optimize=0, exploration_bitstrings=[bits],
    )

    assert len(result.samples_raw) == 1
    row = result.samples_raw[0]
    assert row["target_hit"] is False
    assert row["in_range"] is False
    assert row["valid"] is not score_fails
    if score_fails:
        assert row["invalid_reason"] == "property_failed"
        assert row["sas"] is None
        assert result.trajectory == result.samples_unique == []
        runner._free_energy.assert_not_called()
    else:
        assert row["sas"] == pytest.approx(1.0)
        assert len(result.trajectory) == len(result.samples_unique) == 1
        runner._free_energy.assert_called_once()
