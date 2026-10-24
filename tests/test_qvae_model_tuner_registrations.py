"""Registration contract for the QVAE-MNIST model tuner."""

import importlib
import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_PATH = list(sys.path)
sys.path[:] = [
    entry
    for entry in ORIGINAL_PATH
    if not entry.endswith("qdiffusion") and not entry.endswith("dplm")
]
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

_spec = importlib.util.spec_from_file_location(
    "qvae_model_tuner_registrations",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "trainer", "model_tuner.py"),
)
model_tuner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_tuner)
ModelTuner = model_tuner.ModelTuner
sys.path[:] = ORIGINAL_PATH


def test_single_stage_registrations_are_stored():
    """Model, optimizer and loaders are kept for the single-stage path."""
    tuner = ModelTuner()
    model = object()
    optimizer = object()
    train_loader = object()
    test_loader = object()
    tuner.register_model(model)
    tuner.register_optimiser(optimizer)
    tuner.register_dataLoaders(train_loader, test_loader)
    assert tuner._model is model
    assert tuner._optimiser is optimizer
    assert tuner.train_loader is train_loader
    assert tuner.test_loader is test_loader
    assert tuner._use_two_optimisers is False


def test_two_stage_registration_flips_the_mode_flag():
    """Registering VAE/BM optimizers switches to the two-stage path."""
    tuner = ModelTuner()
    vae_optimizer = object()
    bm_optimizer = object()
    tuner.register_optimiser(object())
    tuner.register_two_optimisers(vae_optimizer, bm_optimizer)
    assert tuner._vae_optimiser is vae_optimizer
    assert tuner._bm_optimiser is bm_optimizer
    assert tuner._use_two_optimisers is True
