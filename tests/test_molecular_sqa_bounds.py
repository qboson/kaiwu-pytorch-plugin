"""Check coefficient bounds before building the matched SQA energy model."""
import importlib
import itertools
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("dimod")


@pytest.fixture
def solver(monkeypatch):
    prefix = "qc_mol_design"
    saved = {k: v for k, v in sys.modules.items()
             if k == prefix or k.startswith(prefix + ".")}
    for name in saved:
        sys.modules.pop(name)
    root = Path(__file__).resolve().parents[1] / "example/molecular_design/sqa"
    monkeypatch.syspath_prepend(str(root))
    try:
        with warnings.catch_warnings():
            yield importlib.import_module("qc_mol_design.optimization.sqa_solver")
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_rejects_signed_minimum_without_abs_overflow(solver, dtype):
    value = np.iinfo(dtype).min
    matrix = np.array([[0, value], [value, 0]], dtype=dtype)
    with pytest.raises(ValueError, match="14-bit"):
        solver.energy_model(matrix)


@pytest.mark.parametrize("value", [-8192, 8192, 1.5, np.inf, np.nan])
def test_rejects_other_invalid_coefficients(solver, value):
    with pytest.raises(ValueError, match="14-bit"):
        solver.energy_model(np.array([[0., value], [value, 0.]]))


@pytest.mark.parametrize("dtype", [np.int16, np.int64, np.float64])
def test_boundary_coefficients_preserve_energy(solver, dtype):
    matrix = np.array([[0, 8191, -8191], [8191, 0, 0], [-8191, 0, 0]], dtype=dtype)
    model = solver.energy_model(matrix)
    for state in itertools.product((-1, 1), repeat=3):
        spins = np.asarray(state, dtype=np.float64)
        assert model.energy(dict(enumerate(state))) == -spins @ matrix @ spins
