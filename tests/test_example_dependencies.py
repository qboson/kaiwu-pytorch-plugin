"""Exercise example dependencies together without downloading data or weights."""

import importlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import anndata
import imageio.v3 as imageio
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PIL import Image
import scanpy as sc
import scib_metrics
from scgraph import scGraph
from scipy import sparse
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision import transforms
from transformers import EsmConfig, EsmTokenizer

import kaiwu as kw
from kaiwu.torch_plugin import BoltzmannMachine


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_example(name, relative_path):
    """Load an example file without colliding with other examples' modules."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_example_dependencies():
    """Image transforms, PNG I/O, plotting, and metrics agree on pixel values."""
    pixels = np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    encoded = imageio.imwrite("<bytes>", pixels, extension=".png")
    image = Image.open(io.BytesIO(encoded))
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    other = (tensor + 0.1).clamp(max=1)
    expected = -10 * torch.log10((tensor - other).square().mean())
    torch.testing.assert_close(PeakSignalNoiseRatio(data_range=1.0)(tensor, other), expected)
    np.testing.assert_array_equal(imageio.imread(encoded), pixels)

    figure = Figure()
    sns.heatmap(pd.DataFrame(pixels.mean(axis=2)), ax=figure.subplots())
    output = io.BytesIO()
    figure.savefig(output, format="png")
    assert Image.open(output).size[0] > 0
    _load_example("mnist_dependency_helpers", "example/qvae_mnist/utils/helpers.py")


def test_cell_analysis_dependencies(tmp_path):
    """AnnData, Scanpy, scIB, and scGraph process the same small cell dataset."""
    rng = np.random.default_rng(42)
    labels = np.tile(np.repeat(np.arange(4), 16), 2)
    matrix = rng.lognormal(mean=1, sigma=0.5, size=(4, 64))[labels]
    matrix += rng.normal(scale=0.1, size=(128, 64))
    observations = pd.DataFrame(
        {"batch": np.repeat(["a", "b"], 64), "label": labels.astype(str)},
        index=[f"cell_{index}" for index in range(128)],
    )
    adata = anndata.AnnData(sparse.csr_matrix(matrix), obs=observations)
    adata.obsm["X_qvae"] = PCA(n_components=3).fit_transform(matrix)
    path = tmp_path / "cells.h5ad"
    adata.write_h5ad(path)
    restored = anndata.read_h5ad(path)
    sc.pp.neighbors(restored, n_neighbors=5, use_rep="X_qvae")
    sc.tl.leiden(restored, flavor="igraph", n_iterations=2, directed=False)
    assert restored.obs["leiden"].nunique() > 1
    classifier = LogisticRegression().fit(restored.obsm["X_qvae"], labels)
    assert classifier.score(restored.obsm["X_qvae"], labels) == 1
    assert np.isfinite(scib_metrics.silhouette_label(restored.obsm["X_qvae"], labels))

    result = scGraph(
        str(path), batch_key="batch", label_key="label", only_umap=False,
        thres_batch=16, thres_celltype=8,
    ).main()
    assert len(result) == 1
    assert np.isfinite(result.to_numpy(dtype=float)).all()


def test_cell_qvae_training(monkeypatch):
    """The real cell trainer records finite losses and saves updated weights."""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "example/qvae_cell"))
    module = _load_example("cell_dependency_models", "example/qvae_cell/models.py")
    trainer_module = _load_example("cell_dependency_trainer", "example/qvae_cell/trainer.py")
    config = SimpleNamespace(
        num_latent_units=4, loss_type="mse", dist_beta=1.0,
        kl_beta=1e-6, weight_decay=0.0,
    )
    model = module.CellQVAE(
        input_dimension=8, activation_fct=None, config=config,
        encoder=module.QVAEEncoder(8, 16, 4),
        decoder=module.QVAEDecoder(6, 16, 8),
        bm=BoltzmannMachine(4, device="cpu"),
        sampler=kw.classical.BruteForceOptimizer(), n_batches=2,
    )
    data = torch.randn(8, 8)
    batch = torch.arange(8) % 2
    loader = DataLoader(TensorDataset(data, batch), batch_size=4)
    trainer = trainer_module.Trainer(SimpleNamespace(bm_weight_decay=0.0), "cpu")
    vae_optimizer = torch.optim.SGD(
        list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=0.01,
    )
    bm_optimizer = torch.optim.SGD(model.bm.parameters(), lr=0.01)
    previous = model.encoder.fc1.weight.detach().clone()
    for train in (True, False):
        metrics = trainer.run_epoch(model, loader, vae_optimizer, bm_optimizer, train)
        assert set(metrics) == set(trainer.metric_keys)
        assert np.isfinite(list(metrics.values())).all()
    assert not torch.equal(previous, model.encoder.fc1.weight)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))


def test_dplm_transformer_training(monkeypatch, tmp_path):
    """The real patched ESM model trains with a local tokenizer and tiny config."""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "example/qdiffusion"))
    module = importlib.import_module("dplm.models.esm_patch")
    tokens = ["<cls>", "<pad>", "<eos>", "<unk>", "A", "C", "D", "X", "<mask>"]
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    tokenizer = EsmTokenizer(vocab_file=str(vocabulary))
    tokenizer.save_pretrained(tmp_path)
    config = EsmConfig(
        vocab_size=len(tokens), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
        pad_token_id=1, mask_token_id=8, token_dropout=False,
        position_embedding_type="rotary",
    )
    config._name_or_path = str(tmp_path)  # pylint: disable=protected-access
    model = module._EsmForDPLM(config, dropout=0)  # pylint: disable=protected-access
    inputs = tokenizer(["ACDX", "DCA"], padding=True, return_tensors="pt")["input_ids"]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    previous = model.lm_head.bias.detach().clone()
    output = model(inputs)
    assert output["logits"].shape == (*inputs.shape, len(tokens))
    loss = torch.nn.functional.cross_entropy(
        output["logits"].flatten(0, 1), inputs.flatten(), ignore_index=1,
    )
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(previous, model.lm_head.bias)
