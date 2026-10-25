"""Contract test for ``example/qvae_mnist/model/config.py::Config.__repr__``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "example" / "qvae_mnist" / "model" / "config.py"


def _load_config():
    spec = importlib.util.spec_from_file_location("qvae_mnist_config_uut", CONFIG)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_repr_exposes_the_public_configuration():
    config_module = _load_config()
    config = config_module.Config()

    text = repr(config)

    assert text.startswith("Config({")
    assert text.endswith("})")
    assert "'type': 'QVAE'" in text
    assert "'num_latent_units': 256" in text
    assert "'dist_beta': 10.0" in text
    assert "'lr': 0.0008" in text
    assert "activation_fct" in text


def test_repr_tracks_kwarg_overrides():
    config_module = _load_config()

    custom_text = repr(config_module.Config(lr=0.1, batch_size=8))

    assert "'lr': 0.1" in custom_text
    assert "'batch_size': 8" in custom_text
