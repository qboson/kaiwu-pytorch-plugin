"""Single-cell QVAE can use a selected layer without modifying the input matrices."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

anndata = pytest.importorskip("anndata")
pytest.importorskip("scanpy")
pytest.importorskip("seaborn")


@pytest.fixture
def cli(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "example" / "qvae_cell"
    monkeypatch.syspath_prepend(str(root))
    for name in ("train_qvae_cell", "trainer", "models", "visualization"):
        monkeypatch.setitem(sys.modules, name, None)
        sys.modules.pop(name)
    return importlib.import_module("train_qvae_cell")


def test_cli_exports_layer_representation_and_source_metadata(cli, monkeypatch, tmp_path):
    data = anndata.AnnData(np.zeros((6, 2), dtype=np.float32))
    data.obs["batch"] = "one"
    data.layers["log1p"] = csr_matrix(np.arange(12, dtype=np.float32).reshape(6, 2))
    source = tmp_path / "input.h5ad"
    data.write_h5ad(source)
    monkeypatch.setattr(sys, "argv", [
        "train_qvae_cell.py", "--data-path", str(source), "--output-dir", str(tmp_path),
        "--input-layer", "log1p", "--device", "cpu",
    ])
    monkeypatch.setattr(cli.Trainer, "create_model", lambda *args: (None, None, None))
    monkeypatch.setattr(cli.Trainer, "train", lambda *args: {})
    # Inspect the real evaluation DataLoader independently of the latent model.
    monkeypatch.setattr(cli.Trainer, "extract_representation", lambda self, model, loader: torch.cat(
        [values for values, _ in loader]
    ).numpy())
    monkeypatch.setattr(
        cli.Trainer, "compute_energy", lambda *args: np.zeros(data.n_obs)
    )
    for name in ("setup_plotting", "plot_training_history", "save_umap_plots", "save_energy_plots"):
        monkeypatch.setattr(cli, name, lambda *args: None)
    monkeypatch.setattr(cli.sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.sc.tl, "umap", lambda *args, **kwargs: None)

    cli.main()

    result = anndata.read_h5ad(tmp_path / "immune_kpp_qvae.h5ad")
    np.testing.assert_array_equal(result.X, data.X)
    np.testing.assert_array_equal(result.obsm["X_qvae"], data.layers["log1p"].toarray())
    assert result.uns["qvae_input"] == {"matrix": "layers", "layer": "log1p"}


def test_missing_layer_fails_before_training(cli):
    trainer = cli.Trainer(SimpleNamespace(input_layer="unknown", loss_type="mse"), torch.device("cpu"))
    with pytest.raises(ValueError, match="unknown.*adata.layers"):
        trainer.adata_to_array(anndata.AnnData(np.ones((2, 2))))
