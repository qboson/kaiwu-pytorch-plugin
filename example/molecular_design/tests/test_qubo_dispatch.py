"""Contracts for solve_qubo's backend dispatch (offline; never needs a CIM license or rdkit)."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_BACKENDS = ("kaiwu", "kaiwu_cim", "neal", "sqa")


def _load_qubo(variant: str):
    # Load qubo.py as a standalone module: optimization/__init__.py pulls in
    # designer.py (rdkit/sklearn), which offline environments may not have.
    # qubo.py itself only needs numpy/os/config, so a direct file load is
    # faithful — it just needs the variant root on sys.path for the config
    # import.
    pkg_root = str(ROOT / variant)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    spec = importlib.util.spec_from_file_location(
        f"qubo_dispatch_{variant}",
        ROOT / variant / "qc_mol_design" / "optimization" / "qubo.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuboDispatch(unittest.TestCase):
    def test_unknown_backend_raises_instead_of_silent_fallback(self):
        for variant in ("cim_sa", "sqa"):
            with self.subTest(variant=variant):
                qubo = _load_qubo(variant)
                with patch.dict(os.environ, {"QC_QUBO_SOLVER": "typo_backend"}):
                    with self.assertRaisesRegex(ValueError, "Unsupported QC_QUBO_SOLVER"):
                        qubo.solve_qubo(np.array([[1.0]]), 1, 1)

    def test_explicit_backends_still_dispatch(self):
        for variant in ("cim_sa", "sqa"):
            for backend in SUPPORTED_BACKENDS:
                with self.subTest(variant=variant, backend=backend):
                    qubo = _load_qubo(variant)
                    solver = f"_solve_{backend}"
                    # _solve_kaiwu/_solve_kaiwu_cim take an extra keyword-only arg.
                    setattr(
                        qubo,
                        solver,
                        lambda *a, _name=backend, **k: [{"from": _name}],
                    )
                    with patch.dict(os.environ, {"QC_QUBO_SOLVER": backend}):
                        self.assertEqual(
                            qubo.solve_qubo(np.array([[1.0]]), 1, 1),
                            [{"from": backend}],
                        )

    def test_fallback_chain_surfaces_real_solver_errors(self):
        for variant in ("cim_sa", "sqa"):
            with self.subTest(variant=variant):
                qubo = _load_qubo(variant)

                def boom(*a, **k):
                    raise ValueError("genuine solver error")

                qubo._solve_sqa = boom
                env = dict(os.environ)
                env.pop("QC_QUBO_SOLVER", None)
                with patch.dict(os.environ, env, clear=True):
                    with self.assertRaisesRegex(ValueError, "genuine solver error"):
                        qubo.solve_qubo(np.array([[1.0]]), 1, 1)

    def test_fallback_chain_still_covers_missing_dependency(self):
        for variant in ("cim_sa", "sqa"):
            with self.subTest(variant=variant):
                qubo = _load_qubo(variant)

                def missing(*a, **k):
                    raise ImportError("neal not installed")

                qubo._solve_sqa = missing
                qubo._solve_neal = lambda *a, **k: [{"from": "neal"}]
                env = dict(os.environ)
                env.pop("QC_QUBO_SOLVER", None)
                with patch.dict(os.environ, env, clear=True):
                    self.assertEqual(
                        qubo.solve_qubo(np.array([[1.0]]), 1, 1),
                        [{"from": "neal"}],
                    )


if __name__ == "__main__":
    unittest.main()
