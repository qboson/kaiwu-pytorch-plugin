"""Empty property-prediction loaders must raise, not return NaN MAE rows."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _install_kaiwu_stubs() -> dict:
    """Provide enough kaiwu.torch_plugin surface for CRBM import without the SDK."""
    saved = {
        name: sys.modules.get(name)
        for name in ("kaiwu", "kaiwu.torch_plugin")
    }

    kaiwu_mod = types.ModuleType("kaiwu")
    torch_plugin = types.ModuleType("kaiwu.torch_plugin")

    class BoltzmannMachine:  # noqa: D401 - import stub only
        pass

    class RestrictedBoltzmannMachine:  # noqa: D401 - import stub only
        pass

    torch_plugin.BoltzmannMachine = BoltzmannMachine
    torch_plugin.RestrictedBoltzmannMachine = RestrictedBoltzmannMachine
    kaiwu_mod.torch_plugin = torch_plugin
    sys.modules["kaiwu"] = kaiwu_mod
    sys.modules["kaiwu.torch_plugin"] = torch_plugin
    return saved


def _restore_modules(saved: dict) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_metrics(backend: str):
    """
    Load one backend's metrics.py without executing evaluation/__init__.py.

    evaluation/__init__.py also imports visualization (RDKit). The empty-loader
    guard does not need that surface, so we register a namespace package instead.
    """
    backend_root = ROOT / "example" / "molecular_design" / backend
    package_root = backend_root / "qc_mol_design"
    metrics_path = package_root / "evaluation" / "metrics.py"

    prefix_names = [
        name
        for name in list(sys.modules)
        if name == "qc_mol_design" or name.startswith("qc_mol_design.")
    ]
    saved_qc = {name: sys.modules.pop(name) for name in prefix_names}

    sys.path.insert(0, str(backend_root))
    try:
        pkg = types.ModuleType("qc_mol_design")
        pkg.__path__ = [str(package_root)]
        sys.modules["qc_mol_design"] = pkg

        evaluation = types.ModuleType("qc_mol_design.evaluation")
        evaluation.__path__ = [str(package_root / "evaluation")]
        sys.modules["qc_mol_design.evaluation"] = evaluation

        # Pull in real submodules used by metrics (config + models).
        importlib.import_module("qc_mol_design.config")
        importlib.import_module("qc_mol_design.models.crbm")
        importlib.import_module("qc_mol_design.models.graph_conv")
        importlib.import_module("qc_mol_design.models.predictors")

        spec = importlib.util.spec_from_file_location(
            "qc_mol_design.evaluation.metrics",
            metrics_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["qc_mol_design.evaluation.metrics"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(backend_root))
        for name in list(sys.modules):
            if name == "qc_mol_design" or name.startswith("qc_mol_design."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_qc)


@pytest.fixture(params=["cim_sa", "sqa"])
def metrics_module(request):
    saved_kaiwu = _install_kaiwu_stubs()
    try:
        yield _load_metrics(request.param)
    finally:
        _restore_modules(saved_kaiwu)


class _EvalOnly:
    """Stand-in for gcn/crbm/pred: only the shared eval() entry is required."""

    def eval(self):
        return self


def test_empty_loader_is_rejected(metrics_module):
    """An empty loader must fail fast instead of averaging zero records into NaN."""
    with pytest.raises(ValueError, match="no batches"):
        metrics_module.compute_prediction_mae(
            _EvalOnly(), _EvalOnly(), _EvalOnly(), []
        )
