"""Custom AnnData columns reach the training covariate and plot calls."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
pytest.importorskip("scanpy")
pytest.importorskip("seaborn")


@pytest.fixture
def cli(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "example" / "qvae_cell"
    monkeypatch.syspath_prepend(str(path))
    for name in ("train_qvae_cell", "trainer", "models", "visualization"):
        monkeypatch.setitem(sys.modules, name, None)
        sys.modules.pop(name)
    return importlib.import_module("train_qvae_cell")


def test_custom_observation_columns_are_used_by_the_cli(cli, monkeypatch, tmp_path):
    data = anndata.AnnData(np.ones((6, 3), dtype=np.float32))
    data.obs["donor"] = ["a", "a", "a", "b", "b", "b"]
    data.obs["cell_type"] = "naive"
    source = tmp_path / "custom.h5ad"
    data.write_h5ad(source)
    saved = {}

    def create_model(self, input_dim, train_loader, **kwargs):
        saved["batches"] = self.n_batches
        return None, None, None

    def save_umap(adata, label, batch, output_dir):
        saved["columns"] = (label, batch)

    monkeypatch.setattr(sys, "argv", [
        "train_qvae_cell.py", "--data-path", str(source), "--output-dir", str(tmp_path),
        "--batch-key", "donor", "--label-key", "cell_type",
    ])
    monkeypatch.setattr(cli.Trainer, "create_model", create_model)
    monkeypatch.setattr(cli.Trainer, "train", lambda *args: {})
    monkeypatch.setattr(cli.Trainer, "extract_representation", lambda *args: np.ones((6, 2)))
    monkeypatch.setattr(cli.Trainer, "compute_energy", lambda *args: np.zeros(6))
    for name in ("setup_plotting", "plot_training_history", "save_energy_plots"):
        monkeypatch.setattr(cli, name, lambda *args: None)
    monkeypatch.setattr(cli, "save_umap_plots", save_umap)
    monkeypatch.setattr(cli.sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.sc.tl, "umap", lambda *args, **kwargs: None)

    cli.main()

    assert saved == {"batches": 2, "columns": ("cell_type", "donor")}


def test_missing_label_is_reported_before_loading_model(cli, monkeypatch, tmp_path):
    data = anndata.AnnData(np.ones((3, 2), dtype=np.float32))
    data.obs["batch"] = "b"
    source = tmp_path / "custom.h5ad"
    data.write_h5ad(source)
    monkeypatch.setattr(sys, "argv", [
        "train_qvae_cell.py", "--data-path", str(source), "--output-dir", str(tmp_path),
        "--label-key", "annotation",
    ])
    monkeypatch.setattr(cli, "setup_plotting", lambda: None)

    with pytest.raises(ValueError, match="annotation.*--label-key.*adata.obs"):
        cli.main()
