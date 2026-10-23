"""Boltzmann-machine selection contract for the MNIST QVAE."""

import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine  # noqa: E402
from model.model import MnistQVAE  # noqa: E402


def build_model(bm_type, latent=7):
    """Builds only the attributes ``_create_bm`` reads."""
    model = MnistQVAE.__new__(MnistQVAE)
    model._latent_dimensions = latent
    model.config = SimpleNamespace(bm_type=bm_type)
    return model


def test_rbm_splits_the_latent_dimension_in_half():
    """The default prior is an RBM with visible = floor(latent / 2)."""
    bm = build_model("rbm")._create_bm()
    assert isinstance(bm, RestrictedBoltzmannMachine)
    assert (bm.num_visible, bm.num_hidden) == (3, 4)
    assert bm.num_nodes == 7


def test_bm_uses_one_node_per_latent_unit():
    """The full-BM prior keeps the whole latent space."""
    bm = build_model("bm")._create_bm()
    assert isinstance(bm, BoltzmannMachine)
    assert bm.num_nodes == 7


def test_unknown_prior_type_is_rejected():
    """Only ``rbm`` and ``bm`` priors are supported."""
    with pytest.raises(ValueError, match="Unsupported bm type"):
        build_model("dbn")._create_bm()
