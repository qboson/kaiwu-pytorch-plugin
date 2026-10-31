"""Contract test for the ``ModelTuner.__init__`` initial state."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TUNER = REPO / "example" / "qvae_mnist" / "trainer" / "model_tuner.py"


def _load_tuner():
    names = ("utils", "utils.logging", "utils.exception")
    saved = {name: sys.modules.get(name) for name in names}
    utils_module = types.ModuleType("utils")
    utils_module.__path__ = []
    logging_module = types.ModuleType("utils.logging")
    logging_module.get_logger = logging.getLogger
    exception_module = types.ModuleType("utils.exception")
    exception_module.ValueError = ValueError
    exception_module.TypeError = TypeError
    sys.modules.update(
        {
            "utils": utils_module,
            "utils.logging": logging_module,
            "utils.exception": exception_module,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location("qvae_mnist_tuner_init_uut", TUNER)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_starts_without_a_model_optimiser_or_data():
    module = _load_tuner()
    config = object()

    tuner = module.ModelTuner(config=config)

    assert tuner._config is config
    assert tuner._model is None
    assert tuner._optimiser is None
    assert tuner._vae_optimiser is None
    assert tuner._bm_optimiser is None
    assert tuner._use_two_optimisers is False
    assert tuner.train_loader is None
    assert tuner.test_loader is None
    assert tuner.outpath == ""
    assert tuner.infile == ""


def test_config_defaults_to_none():
    module = _load_tuner()

    assert module.ModelTuner()._config is None
