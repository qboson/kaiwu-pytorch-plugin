"""Batch-conditioned decoder-input contract for the single-cell QVAE."""

import os
import sys

import importlib.util

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_cell_models", os.path.join(REPO_ROOT, "example", "qvae_cell", "models.py")
)
models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(models)
CellQVAE = models.CellQVAE


def make_model(n_batches):
    """Builds only the attribute ``_decoder_input`` reads."""
    model = CellQVAE.__new__(CellQVAE)
    model.n_batches = n_batches
    return model


def test_without_batches_the_latent_state_is_returned_unchanged():
    """``n_batches <= 0`` disables conditioning entirely."""
    model = make_model(0)
    zeta = torch.zeros(2, 4)
    assert model._decoder_input(zeta) is zeta


def test_missing_batch_index_is_rejected_when_conditioning_is_on():
    """Conditioned models must receive the per-sample batch index."""
    model = make_model(3)
    with pytest.raises(ValueError, match="batch_idx"):
        model._decoder_input(torch.zeros(2, 4))


def test_batch_index_is_appended_as_a_float_one_hot():
    """The one-hot block follows the latent state and has one column per batch."""
    model = make_model(3)
    zeta = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    conditioned = model._decoder_input(zeta, torch.tensor([0, 2]))
    assert conditioned.shape == (2, 7)
    assert conditioned.dtype == zeta.dtype
    assert torch.equal(conditioned[:, :4], zeta)
    assert conditioned[:, 4:].tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
