"""Dataset-expansion contract for the RBM digits runner."""

import importlib.util
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "rbm_digits"))

_spec = importlib.util.spec_from_file_location(
    "rbm_digits_runner_data",
    os.path.join(REPO_ROOT, "example", "rbm_digits", "rbm_digits.py"),
)
rbm_digits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rbm_digits)
RBMRunner = rbm_digits.RBMRunner


def test_runner_expands_digits_with_four_translations_and_scales():
    """Each digit contributes five samples and the splits stay in [0, 1]."""
    runner = RBMRunner.__new__(RBMRunner)
    X_train, X_test, y_train, y_test = runner.load_data(plot_img=False)
    assert X_train.shape[0] + X_test.shape[0] == 5 * 1797
    assert X_train.shape[1] == 64
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
    for split in (X_train, X_test):
        assert np.isfinite(split).all()
        assert split.min() >= -1e-12
        assert split.max() <= 1.0 + 1e-12
