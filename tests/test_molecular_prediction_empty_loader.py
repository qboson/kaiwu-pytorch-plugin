"""The molecular property MAE must not report NaN for an empty loader."""

import importlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["cim_sa", "sqa"])
def metrics_module(request, monkeypatch):
    """Import one backend's metrics module without leaking the other copy."""
    prefix = "qc_mol_design"
    saved = {name: value for name, value in sys.modules.items()
             if name == prefix or name.startswith(prefix + ".")}
    for name in saved:
        sys.modules.pop(name)
    monkeypatch.syspath_prepend(str(ROOT / "example" / "molecular_design" / request.param))
    try:
        yield importlib.import_module(f"{prefix}.evaluation.metrics")
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)


class _EvalOnly:
    def eval(self):
        return self


def test_empty_loader_is_rejected(metrics_module):
    with pytest.raises(ValueError, match="no batches"):
        metrics_module.compute_prediction_mae(_EvalOnly(), _EvalOnly(), _EvalOnly(), [])
