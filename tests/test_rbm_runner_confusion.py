"""Confusion-matrix artifact contract for the RBM digits runner."""

import importlib.util
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "rbm_digits"))

_spec = importlib.util.spec_from_file_location(
    "rbm_digits_runner_confusion",
    os.path.join(REPO_ROOT, "example", "rbm_digits", "rbm_digits.py"),
)
rbm_digits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rbm_digits)
RBMRunner = rbm_digits.RBMRunner


def test_pdf_is_written_into_the_results_directory(tmp_path, monkeypatch):
    """``save_pdf=True`` creates ``results/rbm_confusion_matrix_<suffix>.pdf``."""
    monkeypatch.chdir(tmp_path)
    runner = RBMRunner.__new__(RBMRunner)
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    runner.plot_confusion_matrix(y_true, y_pred, title_suffix="demo", save_pdf=True)
    target = tmp_path / "results" / "rbm_confusion_matrix_demo.pdf"
    assert target.is_file()
    assert target.stat().st_size > 0
