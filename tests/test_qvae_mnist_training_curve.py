"""Training-curve plot contract for the QVAE-MNIST trainer."""

import importlib
import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

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


def test_training_curve_is_saved_as_png(tmp_path):
    """Both loss curves are plotted into ``qvae_training_curve.png``."""
    trainer = Trainer.__new__(Trainer)
    trainer.output_dir = str(tmp_path)
    trainer.train_losses = [1.0, 0.5]
    trainer.test_losses = [2.0, 1.0]
    trainer.config = SimpleNamespace(type="QVAE")
    trainer._plot_training_curve()
    target = tmp_path / "qvae_training_curve.png"
    try:
        assert target.is_file()
        assert target.stat().st_size > 0
        assert plt.gcf().axes[0].get_title() == "QVAE Training Curve"
    finally:
        plt.close("all")
