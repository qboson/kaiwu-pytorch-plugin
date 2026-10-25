"""t-SNE frame-path contract for the QVAE-MNIST trainer."""

import importlib
import os
import sys

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


def test_frame_path_is_zero_padded_and_rendered_without_showing(monkeypatch, tmp_path):
    """The frame lives under ``temp_tsne_frames`` and is rendered with show=False."""
    calls = {}

    def fake_tsne(**kwargs):
        calls.update(kwargs)
        return (None, None, None)

    monkeypatch.setattr(trainer_module, "t_SNE", fake_tsne)
    trainer = Trainer.__new__(Trainer)
    trainer.model = object()
    trainer.test_loader = object()
    trainer.output_dir = str(tmp_path)
    frame_path = trainer._save_tsne_frame(7)
    assert frame_path.endswith(os.path.join("temp_tsne_frames", "tsne_epoch_007.png"))
    assert os.path.isdir(os.path.join(tmp_path, "temp_tsne_frames"))
    assert calls["epochs"] == 7
    assert calls["show"] is False
    assert calls["save_path"] == frame_path
