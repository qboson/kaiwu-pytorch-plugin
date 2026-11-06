"""Offline regression tests for cim_sa tuning_solver backend dispatch.

designer.py rebinds *every* solve call to ``solve_tuning()``::

    from qc_mol_design.optimization.tuning_solver import solve_tuning as solve_qubo

and run_generation_strict.py writes the chosen comparison arm into
``QC_QUBO_SOLVER`` (default "sqa", plus the "neal" arm, or unset).  The
matched-budget tuning path is only meaningful for the Kaiwu arms, so
solve_tuning() must hand the default / SQA / Neal backends straight back to
qubo.solve_qubo() *before* reading any tuning-only parameter.  Otherwise the
whole molecular design example dies with a KeyError (unset) or ValueError on
the default path.

The module is loaded directly from source with stubbed packages, so no kaiwu
SDK / dimod / dwave / rdkit / sklearn / network is needed.  ``numpy`` is the
only third-party import and Q may be any object because the dispatch stub
never reads it.

To reproduce the RED state against a pristine (unfixed) copy of the file::

    KW_TUNING_SOLVER_PATH=/tmp/orig_tuning_solver.py \
        python3 -m pytest example/molecular_design/tests/test_cim_sa_tuning_dispatch.py -q

By default the checked-in ``tuning_solver.py`` is loaded (GREEN).
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

_DEFAULT_TARGET = (
    Path(__file__).resolve().parents[1]
    / "cim_sa" / "qc_mol_design" / "optimization" / "tuning_solver.py"
)
# Override exists only so the unfixed source can be exercised for RED/GREEN
# verification; normal runs use the checked-in file.
TARGET = Path(os.environ.get("KW_TUNING_SOLVER_PATH", str(_DEFAULT_TARGET)))


class _TrackingEnviron(dict):
    """An ``os.environ`` stand-in that records which keys were read."""

    def __init__(self, source):
        super().__init__(source)
        self.accessed = set()

    def __getitem__(self, key):
        self.accessed.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed.add(key)
        return super().get(key, default)


def _load_module(target, environ):
    # Rebuild the whole stub namespace on every load so each test is isolated.
    for name in list(sys.modules):
        if name == "qc_mol_design" or name.startswith("qc_mol_design."):
            del sys.modules[name]

    config = types.ModuleType("qc_mol_design.config")
    config.CFG = types.SimpleNamespace(seed=12345)

    pkg = types.ModuleType("qc_mol_design")
    pkg.__path__ = [str(target.parents[1])]
    op = types.ModuleType("qc_mol_design.optimization")
    op.__path__ = [str(target.parent)]

    cim_hw = types.ModuleType("qc_mol_design.optimization.cim_hardware")
    cim_hw.initialize_license = lambda *a, **k: None
    cim_hw.solve_hardware = lambda *a, **k: None
    cim_hw.LAST_CALL = None

    calls = []
    sentinel = object()

    qubo = types.ModuleType("qc_mol_design.optimization.qubo")

    def fake_solve_qubo(Q, num_reads, num_sweeps,
                        initial_states=None, cim_prepared=None):
        calls.append((Q, num_reads, num_sweeps, initial_states))
        return sentinel

    qubo.solve_qubo = fake_solve_qubo
    qubo._decode_cim_pool = lambda *a, **k: None

    spin = types.ModuleType("qc_mol_design.optimization.spin_encoding")
    spin.encode_prepared = lambda *a, **k: None
    spin.physical_matrix = lambda *a, **k: None

    for name, module in [
        ("qc_mol_design", pkg),
        ("qc_mol_design.config", config),
        ("qc_mol_design.optimization", op),
        ("qc_mol_design.optimization.cim_hardware", cim_hw),
        ("qc_mol_design.optimization.qubo", qubo),
        ("qc_mol_design.optimization.spin_encoding", spin),
    ]:
        sys.modules[name] = module
    op.cim_hardware = cim_hw
    op.qubo = qubo
    op.spin_encoding = spin

    spec = importlib.util.spec_from_file_location(
        "qc_mol_design.optimization.tuning_solver", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # The module only ever uses os.environ; give it a recording, process-local
    # stand-in instead of the real (shared) os.environ.
    fake_os = types.ModuleType("os")
    fake_os.environ = environ
    fake_os.path = os.path
    mod.os = fake_os
    return mod, calls, sentinel


@pytest.mark.parametrize("solver", [None, "sqa", "neal"],
                         ids=["unset", "sqa", "neal"])
def test_non_tuning_backends_delegate_to_qubo_solver(solver):
    environ = _TrackingEnviron(
        {} if solver is None else {"QC_QUBO_SOLVER": solver})
    mod, calls, sentinel = _load_module(TARGET, environ)

    Q = object()
    initial_states = object()
    result = mod.solve_tuning(
        Q, num_reads=10, num_sweeps=1000,
        initial_states=initial_states, cim_prepared=None)

    # No KeyError/ValueError: the call returns the native qubo solver result.
    assert result is sentinel
    assert calls == [(Q, 10, 1000, initial_states)]
    # The early dispatch must happen before any tuning-only parameter is read.
    assert "QC_SOLVER_REPEATS" not in environ.accessed
    assert "KAIWU_CIM_TASK_CONTEXT" not in environ.accessed


def test_kaiwu_arm_does_not_take_the_early_dispatch():
    environ = _TrackingEnviron({"QC_QUBO_SOLVER": "kaiwu"})
    mod, calls, sentinel = _load_module(TARGET, environ)

    try:
        # A real Kaiwu tuning call needs CIM context / prepared data that the
        # offline stubs cannot provide, so it is expected to raise.  What
        # matters is that the dispatch target (stubbed qubo.solve_qubo) is
        # never reached for a genuine tuning backend.
        mod.solve_tuning(object(), num_reads=10, num_sweeps=1000,
                         cim_prepared=None)
    except Exception:
        pass
    assert calls == []
