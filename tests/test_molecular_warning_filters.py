"""Molecular example imports must respect the caller's warning policy."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("variant", ["cim_sa", "sqa"])
def test_molecular_imports_do_not_suppress_caller_warnings(variant):
    example = Path(__file__).resolve().parents[1] / "example/molecular_design" / variant
    script = """
from importlib import import_module
import warnings

class CallerWarning(UserWarning):
    pass

modules = (
    "data.loader",
    "data.featurizer",
    "baselines.fingerprint_models",
    "evaluation.visualization",
    "training.qc_trainer",
    "optimization.designer",
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", CallerWarning)
    for module in modules:
        import_module("qc_mol_design." + module)
        warnings.warn(module, CallerWarning)
assert [str(w.message) for w in caught if w.category is CallerWarning] == list(modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=example, capture_output=True, text=True,
        check=False, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
