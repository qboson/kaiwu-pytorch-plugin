"""DPT roots refer to cell positions, even when cell names are not unique."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(name="benchmark")
def fixture_benchmark():
    """Load the actual benchmark with its optional single-cell dependencies."""
    pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    path = Path(__file__).resolve().parents[1] / "example/qvae_cell/evaluate_benchmark.py"
    spec = importlib.util.spec_from_file_location("cell_dpt_root_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_cells(benchmark, duplicate_names):
    """Make a small connected trajectory with its first root cell at row 20."""
    positions = np.linspace(0, 4, 40, dtype=np.float32)
    expression = np.column_stack([positions, np.sin(positions), np.cos(positions)])
    cells = benchmark.anndata.AnnData(expression)
    cells.obsm["X_qvae"] = expression.copy()
    cells.obs["cell_type"] = ["other"] * 20 + ["root"] * 20
    names = [f"cell_{index}" for index in range(40)]
    if duplicate_names:
        names[0] = names[20] = "shared_barcode"
    cells.obs_names = names
    return cells


@pytest.mark.parametrize("duplicate_names", [False, True])
def test_dpt_uses_first_matching_cell_position(benchmark, duplicate_names):
    """Real Scanpy results agree with an explicitly positioned root oracle."""
    cells = make_cells(benchmark, duplicate_names)
    original_names = cells.obs_names.copy()
    reference = make_cells(benchmark, False)
    benchmark.sc.pp.neighbors(reference, use_rep="X_qvae", n_neighbors=5)
    reference.uns["iroot"] = 20
    benchmark.sc.tl.diffmap(reference)
    benchmark.sc.tl.dpt(reference)

    actual = benchmark.run_dpt(cells, "X_qvae", "root", "cell_type", 5, "result")

    assert cells.uns["iroot"] == 20
    assert cells.obs.iloc[cells.uns["iroot"]]["cell_type"] == "root"
    assert actual.iloc[20] == 0
    assert actual.iloc[0] > 0
    np.testing.assert_allclose(actual.to_numpy(), reference.obs["dpt_pseudotime"],
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(cells.obs_names, original_names)
    np.testing.assert_array_equal(actual, cells.obs["dpt_pseudotime"])


def test_missing_root_type_still_has_clear_error(benchmark):
    """An absent requested cell type retains the existing explicit error."""
    cells = make_cells(benchmark, False)
    with pytest.raises(ValueError, match="'missing' not found in adata.obs"):
        benchmark.run_dpt(cells, "X_qvae", "missing", "cell_type", 5, "result")
