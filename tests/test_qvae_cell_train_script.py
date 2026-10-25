"""Contract test for the ``train_qvae_cell.main`` orchestration."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO / "example" / "qvae_cell" / "train_qvae_cell.py"


class _FakeModel:
    def __init__(self) -> None:
        self.loaded: list[object] = []

    def load_state_dict(self, state) -> None:
        self.loaded.append(state)


class _FakeTrainer:
    instances: list["_FakeTrainer"] = []

    def __init__(self, args, device) -> None:
        self.args = args
        self.device = device
        self.calls: list[tuple] = []
        self.model = _FakeModel()
        _FakeTrainer.instances.append(self)

    def adata_to_array(self, adata):
        self.calls.append(("adata_to_array", adata.n_vars))
        return np.zeros((3, 4), dtype=np.float32)

    def batch_indices(self, adata, batch_key):
        self.calls.append(("batch_indices", batch_key))
        return np.array([0, 0, 1])

    def make_loaders(self, x, batch_indices):
        self.calls.append(("make_loaders", x.shape, batch_indices.tolist()))
        return "train", "val", "eval"

    def create_model(self, input_dim, train_loader):
        self.calls.append(("create_model", input_dim, train_loader))
        return self.model, "vae-opt", "bm-opt"

    def train(self, model, train_loader, val_loader, vae_opt, bm_opt, checkpoint):
        self.calls.append(("train", checkpoint))
        return {"loss": [1.0], "val_loss": [1.1]}

    def extract_representation(self, model, eval_loader):
        self.calls.append(("extract_representation", eval_loader))
        return np.ones((3, 5), dtype=np.float32)

    def compute_energy(self, model, eval_loader):
        self.calls.append(("compute_energy", eval_loader))
        return np.array([0.1, 0.2, 0.3])


class _FakeAdata:
    def __init__(self) -> None:
        self.obs_names = ["cell0", "cell1", "cell2"]
        self.obs = pd.DataFrame(index=self.obs_names)
        self.obsm: dict[str, object] = {}
        self.uns: dict[str, object] = {}
        self.n_obs = 3
        self.n_vars = 4
        self.made_unique = False
        self.written: Path | None = None

    def obs_names_make_unique(self):
        self.made_unique = True

    def write_h5ad(self, path):
        self.written = Path(path)


def _load_script():
    names = ("anndata", "scanpy", "trainer", "visualization")
    saved = {name: sys.modules.get(name) for name in names}
    state: dict[str, object] = {"seeds": [], "plots": {}, "umap": [], "energy": []}

    anndata_module = types.ModuleType("anndata")
    anndata_module.read_h5ad = lambda path: state["adata"]

    scanpy_module = types.ModuleType("scanpy")
    scanpy_module.pp = types.SimpleNamespace(
        neighbors=lambda adata, **kwargs: state["umap"].append(("neighbors", kwargs))
    )
    scanpy_module.tl = types.SimpleNamespace(
        umap=lambda adata, **kwargs: state["umap"].append(("umap", kwargs))
    )

    trainer_module = types.ModuleType("trainer")
    trainer_module.Trainer = _FakeTrainer
    trainer_module.set_seed = lambda seed: state["seeds"].append(seed)

    visualization_module = types.ModuleType("visualization")
    visualization_module.setup_plotting = lambda: state["plots"].__setitem__(
        "setup", state["plots"].get("setup", 0) + 1
    )
    visualization_module.plot_training_history = (
        lambda history, path: state["plots"].__setitem__(
            "history", (history, path)
        )
    )
    visualization_module.save_umap_plots = (
        lambda adata, labels, batches, out: state["umap"].append(
            ("save", labels, batches, out)
        )
    )
    visualization_module.save_energy_plots = (
        lambda adata, labels, out: state["energy"].append((labels, out))
    )

    sys.modules.update(
        {
            "anndata": anndata_module,
            "scanpy": scanpy_module,
            "trainer": trainer_module,
            "visualization": visualization_module,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "qvae_cell_train_script_uut", TRAIN_SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module, state
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _args(tmp_path, load_weights=False):
    return types.SimpleNamespace(
        dataset="immune",
        data_path=None,
        output_dir=str(tmp_path),
        device="cuda:0",
        seed=7,
        load_weights=load_weights,
    )


@pytest.fixture(autouse=True)
def _clear_trainer_registry():
    _FakeTrainer.instances.clear()
    yield
    _FakeTrainer.instances.clear()


def test_training_path_writes_history_representation_and_plots(tmp_path, monkeypatch):
    module, state = _load_script()
    adata = _FakeAdata()
    state["adata"] = adata
    monkeypatch.setattr(module, "parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    module.main()

    trainer = _FakeTrainer.instances[0]
    assert trainer.device == torch.device("cpu")
    assert state["seeds"] == [7]
    assert state["plots"]["setup"] == 1
    assert ("train", str(tmp_path / "immune_kpp_qvae_best.pth")) in trainer.calls
    assert (tmp_path / "training_history.csv").is_file()
    assert (tmp_path / "X_qvae.npy").is_file()
    assert state["plots"]["history"][0] == {"loss": [1.0], "val_loss": [1.1]}
    assert state["plots"]["history"][1] == str(tmp_path / "training_curves.png")
    assert ("neighbors", {"use_rep": "X_qvae", "n_neighbors": 15}) in state["umap"]
    assert ("umap", {"random_state": 7}) in state["umap"]
    assert ("save", "final_annotation", "batch", str(tmp_path)) in state["umap"]
    assert state["energy"] == [("final_annotation", str(tmp_path))]
    assert np.allclose(adata.obsm["X_qvae"], np.ones((3, 5), dtype=np.float32))
    assert np.allclose(adata.obs["QVAE_Energy"], [0.1, 0.2, 0.3])
    assert adata.written == tmp_path / "immune_kpp_qvae.h5ad"


def test_load_weights_path_skips_training(tmp_path, monkeypatch):
    module, state = _load_script()
    adata = _FakeAdata()
    state["adata"] = adata
    monkeypatch.setattr(module, "parse_args", lambda: _args(tmp_path, load_weights=True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loaded_paths: list[tuple[str, object]] = []
    monkeypatch.setattr(
        module.torch,
        "load",
        lambda path, map_location=None: loaded_paths.append((path, map_location))
        or {"weight": torch.tensor(1.0)},
    )

    module.main()

    trainer = _FakeTrainer.instances[0]
    assert not any(call[0] == "train" for call in trainer.calls)
    weights_path = str(tmp_path / "immune_kpp_qvae_best.pth")
    assert loaded_paths == [(weights_path, torch.device("cpu"))]
    assert len(trainer.model.loaded) == 1
    assert not (tmp_path / "training_history.csv").exists()
    assert adata.written == tmp_path / "immune_kpp_qvae.h5ad"
