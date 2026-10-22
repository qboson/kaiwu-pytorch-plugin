"""Regression test for ``parse_resolutions`` on empty input.

``--resolutions ""`` (or ``","``) produced an empty list, so
``evaluate_clustering`` built an empty DataFrame and died in pandas:

    KeyError: 'ARI'

``parse_metrics`` already rejects empty input with a descriptive ValueError;
resolutions should behave the same way.
"""

import os
import sys
import types

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_DIR = os.path.join(REPO_ROOT, "example", "qvae_cell")


@pytest.fixture()
def benchmark(monkeypatch):
    for name in ("anndata", "scanpy", "scib", "scgraph"):
        module = types.ModuleType(name)
        module.__dict__.setdefault("tl", types.SimpleNamespace())
        module.__dict__.setdefault("pp", types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, name, module)
    sys.path.insert(0, BENCHMARK_DIR)
    sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))
    import evaluate_benchmark

    return evaluate_benchmark


@pytest.mark.parametrize("value", ["", "   ", ",", " , "])
def test_empty_resolutions_are_rejected(benchmark, value):
    with pytest.raises(ValueError) as excinfo:
        benchmark.parse_resolutions(value)

    assert "resolutions" in str(excinfo.value)


def test_resolutions_are_parsed(benchmark):
    assert benchmark.parse_resolutions("0.2, 0.4,0.8") == [0.2, 0.4, 0.8]
