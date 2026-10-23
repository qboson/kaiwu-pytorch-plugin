"""Resolution-parsing contract for the qvae_cell benchmark."""

import importlib.util
import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for name in ("scanpy", "anndata"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_benchmark_resolutions",
    os.path.join(REPO_ROOT, "example", "qvae_cell", "evaluate_benchmark.py"),
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def test_comma_separated_values_are_parsed_to_floats():
    """Whitespace around entries is ignored and order is preserved."""
    assert benchmark.parse_resolutions("0.2,0.4,0.8") == [0.2, 0.4, 0.8]
    assert benchmark.parse_resolutions(" 0.5 , , 0.75 ") == [0.5, 0.75]
    assert benchmark.parse_resolutions("1") == [1.0]


def test_empty_value_yields_an_empty_list():
    """An empty option means no resolutions."""
    assert benchmark.parse_resolutions("") == []
    assert benchmark.parse_resolutions(" , ") == []
