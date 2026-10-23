"""Model-construction contract for the qvae_cell trainer."""

import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_cell"))
pre = importlib.import_module("kaiwu.preprocess")
if not hasattr(pre, "PrecisionReducer"):
    pre.PrecisionReducer = type(
        "PrecisionReducer", (), {"__init__": lambda self, *args, **kwargs: None}
    )

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_trainer_model", os.path.join(REPO_ROOT, "example", "qvae_cell", "trainer.py")
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer
CellQVAE = trainer_module.CellQVAE


def build_trainer(latent_dim=4, n_batches=3, input_dim=5):
    """Builds a trainer whose data-mean and sampler steps are stubbed."""
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        latent_dim=latent_dim,
        num_visible=2,
        num_hidden=2,
        hidden_dim=6,
        normalization_method="layer",
        loss_type="bernoulli",
        dist_beta=10.0,
        kl_beta=1e-6,
        lr=1e-3,
        rbm_lr=5e-4,
    )
    trainer.n_batches = n_batches
    trainer.device = torch.device("cpu")
    trainer.create_sampler = lambda: object()
    trainer.compute_mean_x = lambda loader: np.zeros(input_dim, dtype=np.float32)
    return trainer


def test_model_uses_the_configured_dimensions_and_batch_conditioning():
    """Latent width, batch one-hot width and BM size follow the config."""
    model, vae_optimizer, bm_optimizer = build_trainer().create_model(5, None)
    assert isinstance(model, CellQVAE)
    assert model.encoder.fc1.in_features == 5
    assert model.decoder.fc1.in_features == 4 + 3
    assert model.n_batches == 3
    assert model.bm.num_nodes == 4
    encoder_params = sum(1 for _ in model.encoder.parameters())
    decoder_params = sum(1 for _ in model.decoder.parameters())
    assert len(vae_optimizer.param_groups[0]["params"]) == encoder_params + decoder_params
    assert len(bm_optimizer.param_groups[0]["params"]) == sum(
        1 for _ in model.bm.parameters()
    )


def test_latent_dimension_must_match_the_boltzmann_machine():
    """A latent width that differs from visible+hidden is rejected."""
    with pytest.raises(ValueError, match="latent_dim must equal"):
        build_trainer(latent_dim=5).create_model(5, None)
