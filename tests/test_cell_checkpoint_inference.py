"""Read and use single-cell training weights without creating a solver."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

anndata = pytest.importorskip("anndata")
pytest.importorskip("scanpy")
pytest.importorskip("seaborn")


def test_load_weights_cli_works_offline_with_cim_selected(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1] / "example" / "qvae_cell"
    monkeypatch.syspath_prepend(str(root))
    for module in ("train_qvae_cell", "trainer", "models", "visualization"):
        # Restore bare import names even if they were initially absent.
        monkeypatch.setitem(sys.modules, module, None)
        sys.modules.pop(module)
    cli = importlib.import_module("train_qvae_cell")

    labels = np.repeat(["batch-one", "batch-two"], 4)
    obs = pd.DataFrame({"batch": labels, "final_annotation": labels},
                       index=np.arange(len(labels)).astype(str))
    data = anndata.AnnData(np.linspace(0, 1, 24, dtype=np.float32).reshape(8, 3), obs=obs)
    data_path = tmp_path / "cells.h5ad"
    data.write_h5ad(data_path)
    monkeypatch.setattr(sys, "argv", [
        "train_qvae_cell.py", "--data-path", str(data_path), "--output-dir", str(tmp_path),
        "--device", "cpu", "--batch-size", "4", "--latent-dim", "2", "--hidden-dim", "4",
        "--num-visible", "1", "--num-hidden", "1",
        "--sampler-type", "cim", "--load-weights",
    ])
    args = cli.parse_args()
    trainer = cli.Trainer(args, torch.device("cpu"))
    batch_ids = trainer.batch_indices(data, "batch")
    train_loader, _, eval_loader = trainer.make_loaders(trainer.adata_to_array(data), batch_ids)
    monkeypatch.setattr(cli.Trainer, "create_sampler", lambda self: object())
    model, _, _ = trainer.create_model(data.n_vars, train_loader)
    expected = trainer.extract_representation(model, eval_loader)
    torch.save(model.state_dict(), tmp_path / "immune_kpp_qvae_best.pth")

    def unexpected(*args, **kwargs):
        pytest.fail("Checkpoint inference must not create a solver or an optimizer")

    monkeypatch.setattr(cli.Trainer, "create_sampler", unexpected)
    monkeypatch.setattr(torch.optim, "Adam", unexpected)
    # Visualization is independent of checkpoint loading and representation export.
    for name in ("setup_plotting", "save_umap_plots", "save_energy_plots"):
        monkeypatch.setattr(cli, name, lambda *args: None)
    monkeypatch.setattr(cli.sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.sc.tl, "umap", lambda *args, **kwargs: None)

    cli.main()

    output = anndata.read_h5ad(tmp_path / "immune_kpp_qvae.h5ad")
    np.testing.assert_allclose(output.obsm["X_qvae"], expected)
    assert np.isfinite(output.obs["QVAE_Energy"]).all()
