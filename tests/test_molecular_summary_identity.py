"""Generation summaries count molecules rather than SMILES spellings."""
import importlib
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")

@pytest.fixture(params=["cim_sa", "sqa"])
def metrics(request, monkeypatch):
    prefix = "qc_mol_design"
    saved = {k: v for k, v in sys.modules.items()
             if k == prefix or k.startswith(prefix + ".")}
    for name in saved:
        sys.modules.pop(name)
    root = Path(__file__).resolve().parents[1] / "example/molecular_design" / request.param
    monkeypatch.syspath_prepend(str(root))
    try:
        with warnings.catch_warnings():
            module = importlib.import_module("qc_mol_design.evaluation.metrics")
            yield module
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)




@pytest.mark.parametrize("smiles,expected", [
    (["CCO", "OCC"], 1),
    (["CCO", "OCC", "CCN"], 2),
    (["F[C@H](Cl)Br", "F[C@@H](Cl)Br"], 2),
    (["CCO", "CCO"], 1),
])
def test_summary_counts_molecular_identities(metrics, smiles, expected):
    rows = [dict(smiles=s, valid=True, target_hit=True, qed=.5, logp=1., sas=2.)
            for s in smiles]
    rows.append(dict(smiles="", valid=False, target_hit=False))
    train = pd.DataFrame(dict(qed=[.5], logp=[1.], sas=[2.]))
    stats = metrics.summarise_generation(rows, "qed", (0., 1.), train)
    quality = metrics.generation_quality_metrics(rows, set())
    assert stats["n_unique_valid"] == expected
    assert stats["n_unique_valid"] == quality["n_unique_valid"]
    assert stats["n_duplicates"] == len(smiles) - expected
    assert stats["uniqueness_pct"] == pytest.approx(100 * expected / len(smiles))
    assert stats["n_sampled"] == len(rows)
    assert stats["n_valid"] == len(smiles)
    assert stats["n_in_range"] == len(smiles)
    assert stats["success_all_pct"] == pytest.approx(100 * len(smiles) / len(rows))

@pytest.mark.parametrize("rows", [[], [dict(smiles="CCO", valid=False)]])
def test_summary_without_valid_reads(metrics, rows):
    train = pd.DataFrame(dict(qed=[.5], logp=[1.], sas=[2.]))
    stats = metrics.summarise_generation(rows, "qed", (0., 1.), train)
    assert stats["n_sampled"] == len(rows)
    assert stats["n_valid"] == 0
    assert stats["n_unique_valid"] == 0
    assert stats["n_duplicates"] == 0
    assert stats["uniqueness_pct"] == 0
