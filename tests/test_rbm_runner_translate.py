"""Image-shift contract for the RBM digits runner method."""

import importlib.util
import os

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "rbm_digits_runner_method",
    os.path.join(REPO_ROOT, "example", "rbm_digits", "rbm_digits.py"),
)
rbm_digits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rbm_digits)
RBMRunner = rbm_digits.RBMRunner


def center_image():
    """One 3x3 image with a single lit centre pixel."""
    image = np.zeros((3, 3), dtype=np.float32)
    image[1, 1] = 1.0
    return image


def test_method_shifts_the_pixel_in_each_direction():
    """The runner method mirrors the standalone shift helper."""
    runner = RBMRunner.__new__(RBMRunner)
    expected = {
        "up": (0, 1),
        "down": (2, 1),
        "left": (1, 0),
        "right": (1, 2),
    }
    for direction, (row, column) in expected.items():
        shifted = runner.translate_image(center_image(), direction)
        assert shifted[row, column] == pytest.approx(1.0)
        assert shifted.sum() == pytest.approx(1.0)


def test_method_rejects_unknown_directions():
    """Only the four documented directions are accepted."""
    runner = RBMRunner.__new__(RBMRunner)
    with pytest.raises(ValueError, match="Invalid direction"):
        runner.translate_image(center_image(), "diagonal")
