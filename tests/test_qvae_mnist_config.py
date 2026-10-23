"""Configuration contract for the QVAE-MNIST example."""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

from model.config import Config  # noqa: E402


def test_default_qvae_settings():
    """The QVAE branch keeps its RBM/Bernoulli defaults and pipeline budgets."""
    config = Config()
    assert config.type == "QVAE"
    assert config.bm_type == "rbm"
    assert config.loss_type == "bernoulli"
    assert config.sampler_type == "sa"
    assert config.num_latent_units == 256
    assert config.encoder_hidden_nodes == [512]
    assert config.decoder_hidden_nodes == [512]
    assert config.dist_beta == 10.0
    assert config.kl_beta == pytest.approx(1e-6)
    assert config.weight_decay == pytest.approx(0.01)
    assert config.batch_size == 256
    assert config.num_epochs == 50
    assert config.lr == pytest.approx(8e-4)
    assert config.use_cuda is False
    assert config.name == "mnist"
    assert config.classifier_kwargs == {}


def test_cell_qvae_settings():
    """The CellQVAE branch uses a full BM, a larger KL weight and layer norm."""
    config = Config("CellQVAE")
    assert config.type == "CellQVAE"
    assert config.bm_type == "bm"
    assert config.kl_beta == pytest.approx(1e-5)
    assert config.hidden_dim == 512
    assert config.normalization_method == "layer"


def test_kwargs_only_override_existing_attributes():
    """Known attributes can be overridden; unknown kwargs are ignored."""
    config = Config(
        "QVAE",
        num_epochs=3,
        bm_type="bm",
        output_dir="runs",
        classifier_kwargs={"alpha": 1},
        unknown_option="ignored",
    )
    assert config.num_epochs == 3
    assert config.bm_type == "bm"
    assert config.output_dir == "runs"
    assert config.classifier_kwargs == {"alpha": 1}
    assert not hasattr(config, "unknown_option")


def test_unknown_model_type_is_rejected():
    """Only the two example families are supported."""
    with pytest.raises(ValueError, match="Unsupported model type"):
        Config("Transformer")


def test_repr_lists_the_public_attributes():
    """The repr starts with the class name and contains a known setting."""
    text = repr(Config("QVAE", num_epochs=2))
    assert text.startswith("Config(")
    assert "'num_epochs': 2" in text
