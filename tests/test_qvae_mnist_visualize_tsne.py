"""t-SNE delegation contract for the QVAE-MNIST trainer."""

import importlib
import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_PATH = list(sys.path)
sys.path[:] = [p for p in ORIGINAL_PATH if not p.endswith("qdiffusion") and not p.endswith("dplm")]
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]
for _name in [m for m in list(sys.modules) if m == "trainer" or m.startswith("trainer.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

trainer_module = importlib.import_module("trainer.trainer")
sys.path[:] = ORIGINAL_PATH
Trainer = trainer_module.Trainer


def test_default_save_path_and_arguments_are_forwarded(monkeypatch, tmp_path):
    """The method derives the default path and forwards every t-SNE argument."""
    calls = {}

    def fake_tsne(**kwargs):
        calls.update(kwargs)
        return ("df", "path", "status")

    monkeypatch.setattr(trainer_module, "t_SNE", fake_tsne)
    trainer = Trainer.__new__(Trainer)
    trainer.model = object()
    trainer.test_loader = object()
    trainer.output_dir = str(tmp_path)
    trainer.config = SimpleNamespace(type="QVAE", num_epochs=3)
    result = trainer._visualize_tsne(point_size=10, alpha=0.4, show=False)
    assert result == ("df", "path", "status")
    assert calls["point_size"] == 10
    assert calls["alpha"] == 0.4
    assert calls["epochs"] == 3
    assert calls["show"] is False
    assert calls["save_path"].endswith("QVAE_t-SNE_epochs_3.png")


def test_untrained_model_is_rejected():
    """Calling before training raises a clear error."""
    trainer = Trainer.__new__(Trainer)
    trainer.model = None
    # The example custom ValueError currently drops its message (see #392).
    with pytest.raises(ValueError):
        trainer._visualize_tsne()
