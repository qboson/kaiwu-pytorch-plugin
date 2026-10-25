"""Entrypoint-export contract for the DPLM ESM2 evaluation wrapper."""

import importlib
import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion"))

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]

module = importlib.import_module("dplm.eval_esm2_distances")
evaluation = importlib.import_module("dplm.workflows.esm2_eval")


def test_main_is_reexported_from_the_evaluation_module():
    """The compatibility entrypoint runs the real evaluation main."""
    assert module.main is evaluation.main


def test_public_configuration_names_are_reexported():
    """``import *`` keeps the evaluation dataclasses available."""
    for name in ("EvalConfig", "GenerationConfig"):
        assert hasattr(module, name), name
        assert getattr(module, name) is getattr(evaluation, name)
