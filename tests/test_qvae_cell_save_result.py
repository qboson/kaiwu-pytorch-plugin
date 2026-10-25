"""CSV-output contract for the qvae_cell benchmark."""

import importlib.util
import os
import sys
import types

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for name in ("scanpy", "anndata"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_benchmark_save", os.path.join(REPO_ROOT, "example", "qvae_cell", "evaluate_benchmark.py")
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def test_result_table_is_written_without_the_index(tmp_path, capsys):
    """The CSV keeps only the DataFrame columns and prints the destination."""
    table = pd.DataFrame({"metric": ["ARI", "NMI"], "value": [0.8, 0.9]})
    target = tmp_path / "metrics.csv"
    benchmark.save_result(table, str(target), "clustering")
    assert pd.read_csv(target).to_dict("list") == {
        "metric": ["ARI", "NMI"],
        "value": [0.8, 0.9],
    }
    assert f"Saved clustering: {target}" in capsys.readouterr().out
