# -*- coding: utf-8 -*-
"""Display-flag tests for the qvae_mnist plotting helper.

The helper module imports optional example dependencies (pandas, tqdm, gif,
imageio) that are not part of the test environment; they are stubbed before
the module is loaded from its example path.
"""

import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402  (backend must be selected before pyplot)
matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER_PATH = os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py")


def _make_stub(name):
    """Empty module stand-in with a valid spec (find_spec-safe)."""
    stub = types.ModuleType(name)
    stub.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return stub


def _load_helpers():
    for name in ("pandas", "tqdm", "gif", "imageio"):
        sys.modules.setdefault(name, _make_stub(name))
    sys.modules["tqdm"].tqdm = lambda it, *a, **k: it
    sys.modules["gif"].frame = lambda func: func
    sys.modules["gif"].options = lambda **k: None
    spec = importlib.util.spec_from_file_location("kpp_helpers_under_test", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPlotTrainingCurvesShowFlag(unittest.TestCase):
    """show=False must not open a figure window; show=True shows it once."""

    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_helpers()

    def test_show_false_displays_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "curves.png")
            with mock.patch("matplotlib.pyplot.show") as show:
                self.helpers.plot_training_curves(
                    [1.0, 0.8], [0.9, 0.7], [50, 80], [45, 75],
                    save_path=save_path, show=False,
                )
            show.assert_not_called()
            self.assertTrue(os.path.exists(save_path))

    def test_show_true_displays_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "curves.png")
            with mock.patch("matplotlib.pyplot.show") as show:
                self.helpers.plot_training_curves(
                    [1.0, 0.8], [0.9, 0.7], [50, 80], [45, 75],
                    save_path=save_path, show=True,
                )
            show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
