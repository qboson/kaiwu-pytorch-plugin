"""Regression test for DPT pseudotime counting in ``evaluate_benchmark``.

``scanpy.tl.dpt`` assigns ``np.inf`` (not NaN) to cells that are not reachable
from the root cell's graph component.  ``evaluate_dpt`` filtered with
``.notna()``, so unreachable cells were counted in ``n_cells_with_dpt`` and
their infinite pseudotime values entered the Kendall-tau comparison against the
PCA baseline — silently wrong trajectory metrics.

scanpy/anndata are optional example dependencies, so they are stubbed here; the
code under test is the counting/masking logic.
"""

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_DIR = os.path.join(REPO_ROOT, "example", "qvae_cell")


class _FakeAnnData:
    def __init__(self, names, obsm=None):
        self.obs_names = pd.Index(names)
        self.obs = pd.DataFrame(index=self.obs_names)
        self.obsm = dict(obsm or {})

    def copy(self):
        return _FakeAnnData(list(self.obs_names), self.obsm)


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


def test_unreachable_cells_are_not_counted_as_valid(benchmark):
    names = ["a", "b", "c", "d"]

    def fake_run_dpt(adata, rep_key, root_cell_type, label_key, n_neighbors, out_key):
        # b is unreachable from the root (scanpy marks it inf), c is missing
        return pd.Series([0.1, np.inf, np.nan, 0.4], index=adata.obs_names)

    benchmark.run_dpt = fake_run_dpt

    adata = _FakeAnnData(names, obsm={"X_qvae": np.zeros((4, 2)), "X_pca": np.zeros((4, 2))})
    row = benchmark.evaluate_dpt(adata, "X_qvae", "label", "root", 15).iloc[0]

    assert row["n_cells_with_dpt"] == 2, (
        "only cells with finite pseudotime are usable DPT cells"
    )
    assert np.isfinite(row["kendall_tau_vs_X_pca"])
