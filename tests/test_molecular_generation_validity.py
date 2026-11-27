"""Both molecular design variants should count raw read failures consistently."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("rdkit")


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_parseable_smiles_with_failed_properties_stays_invalid(variant, monkeypatch):
    root = Path.cwd() / "example" / "molecular_design" / variant
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        f"{variant}_evaluation_metrics", root / "qc_mol_design" / "evaluation" / "metrics.py"
    )
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)

    quality = metrics.generation_quality_metrics(
        [
            {"smiles": "CC", "valid": False},
            {"smiles": "CCO", "valid": True},
            {"smiles": "OCC", "valid": True},
            {"smiles": "N"},  # Older records can omit the validity flag.
        ],
        {"CCO"},
    )

    assert quality["n_sampled"] == 4
    assert quality["n_valid"] == 3
    assert quality["n_unique_valid"] == 2
    assert quality["validity"] == pytest.approx(3 / 4)
    assert quality["uniqueness"] == pytest.approx(2 / 3)
    assert quality["novelty"] == pytest.approx(1 / 2)
