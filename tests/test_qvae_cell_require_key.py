"""Required-key validation contract for the qvae_cell benchmark."""

import importlib.util
import os
import sys
import types

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for name in ("scanpy", "anndata"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_benchmark_require_key",
    os.path.join(REPO_ROOT, "example", "qvae_cell", "evaluate_benchmark.py"),
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def test_present_keys_pass_silently():
    """A present key returns None."""
    assert benchmark.require_key({"leiden": [0, 1]}, "leiden", "adata.obs") is None


def test_missing_key_is_reported_with_container_and_name():
    """The error names the key and the container it was searched in."""
    with pytest.raises(KeyError, match="'leiden' not found in adata.obs"):
        benchmark.require_key({"other": 1}, "leiden", "adata.obs")
