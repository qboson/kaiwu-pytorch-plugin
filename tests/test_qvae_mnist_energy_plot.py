"""Energy-distribution plot contract for the QVAE-MNIST trainer."""

import importlib
import os
import sys

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


def test_energy_plot_is_written_next_to_the_training_output(tmp_path):
    """The violin plot is saved as ``qvae_energy_by_class.png`` at 300 dpi."""
    trainer = Trainer.__new__(Trainer)
    energies = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    labels = [0, 0, 1, 1, 2, 2]
    trainer._save_energy_plot(energies, labels, str(tmp_path))
    target = tmp_path / "qvae_energy_by_class.png"
    try:
        assert target.is_file()
        assert target.stat().st_size > 0
    finally:
        plt.close("all")
