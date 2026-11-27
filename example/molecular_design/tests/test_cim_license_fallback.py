"""Regression tests for the CIM license-cache fallback.

initialize_license() is supposed to fall back to kw.license.init() whenever
the cached license cannot be read or validated.  These tests load the two
example copies directly with stubbed kaiwu modules, so no SDK / dimod /
network is needed.
"""

import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

MD = Path(__file__).resolve().parents[1]
VARIANTS = [
    MD / "cim_sa" / "qc_mol_design" / "optimization" / "cim_hardware.py",
    MD / "sqa" / "qc_mol_design" / "optimization" / "cim_hardware.py",
]
IDS = ["cim_sa", "sqa"]
CREDENTIALS = '{"user_id":"u1","sdk_code":"c1"}'


def _load(path):
    for name in list(sys.modules):
        if name.split(".")[0] in ("kaiwu", "qc_mol_design"):
            del sys.modules[name]

    kw = types.ModuleType("kaiwu")
    lic = types.ModuleType("kaiwu.license")
    cim = types.ModuleType("kaiwu.cim")
    lu = types.ModuleType("kaiwu.license._license_utils")
    ad = types.ModuleType("kaiwu.cim._optimizer_adapter")
    calls = []
    lic.init = lambda *a, **k: calls.append(a)
    lic._license_utils = lu
    cim._optimizer_adapter = ad
    kw.license = lic
    kw.cim = cim

    pkg = types.ModuleType("qc_mol_design")
    pkg.__path__ = [str(path.parents[1])]
    op = types.ModuleType("qc_mol_design.optimization")
    op.__path__ = [str(path.parent)]
    se = types.ModuleType("qc_mol_design.optimization.spin_encoding")
    se.physical_matrix = lambda *a, **k: None
    se.logical_spins = lambda *a, **k: None
    op.spin_encoding = se

    for name, module in [
        ("kaiwu", kw),
        ("kaiwu.license", lic),
        ("kaiwu.cim", cim),
        ("kaiwu.license._license_utils", lu),
        ("kaiwu.cim._optimizer_adapter", ad),
        ("qc_mol_design", pkg),
        ("qc_mol_design.optimization", op),
        ("qc_mol_design.optimization.spin_encoding", se),
    ]:
        sys.modules[name] = module

    mod = types.ModuleType("qc_mol_design.optimization.cim_hardware")
    mod.__package__ = "qc_mol_design.optimization"
    sys.modules[mod.__name__] = mod
    op.cim_hardware = mod
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
    mod.__dict__["_LICENSE_READY"] = False
    return mod, lu, calls


def _run(path, reader):
    mod, lu, calls = _load(path)
    lu.LICENSE_FILE_PATH = ""
    lu._read_license = reader
    os.environ["KAIWU_CREDENTIALS_FILE"] = "/tmp/kaiwu_credentials.json"
    with mock.patch.object(Path, "read_text", lambda self, *a, **k: CREDENTIALS), \
         mock.patch.object(Path, "chmod", lambda self, mode: None):
        mod.initialize_license()
    return len(calls)


@pytest.mark.parametrize("path", VARIANTS, ids=IDS)
def test_cached_license_missing_user_id_falls_back_to_init(path):
    # A stale/incomplete cache raises KeyError; it must be treated as a miss.
    assert _run(path, lambda *a, **k: ({}, "c1")) == 1


@pytest.mark.parametrize("path", VARIANTS, ids=IDS)
def test_cached_license_non_mapping_payload_falls_back_to_init(path):
    assert _run(path, lambda *a, **k: (None, "c1")) == 1


@pytest.mark.parametrize("path", VARIANTS, ids=IDS)
def test_unreadable_cached_license_still_falls_back_to_init(path):
    def unreadable(*a, **k):
        raise ValueError("corrupt cache")

    assert _run(path, unreadable) == 1


@pytest.mark.parametrize("path", VARIANTS, ids=IDS)
def test_matching_cached_license_does_not_reinitialize(path):
    assert _run(path, lambda *a, **k: ({"user_id": "u1"}, "c1")) == 0
