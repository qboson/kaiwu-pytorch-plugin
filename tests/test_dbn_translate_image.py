"""Image-shift contract for the DBN digits example."""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from supervised_dbn_digits import translate_image  # noqa: E402


def center_image():
    """One 3x3 image with a single lit centre pixel."""
    image = np.zeros((3, 3), dtype=np.float32)
    image[1, 1] = 1.0
    return image


def test_each_direction_moves_the_pixel_by_one_step():
    """The four directions shift the lit pixel to the expected neighbour."""
    expected = {
        "up": (0, 1),
        "down": (2, 1),
        "left": (1, 0),
        "right": (1, 2),
    }
    for direction, (row, column) in expected.items():
        shifted = translate_image(center_image(), direction)
        assert shifted.shape == (3, 3)
        assert shifted[row, column] == pytest.approx(1.0)
        assert shifted.sum() == pytest.approx(1.0)


def test_shifted_pixels_leave_zero_fill_behind():
    """Vacated positions are filled with zeros rather than wrapped around."""
    shifted = translate_image(center_image(), "down")
    assert shifted[1, 1] == pytest.approx(0.0)
    assert shifted[0].sum() == pytest.approx(0.0)


def test_unknown_direction_is_rejected():
    """Only the four documented directions are accepted."""
    with pytest.raises(ValueError, match="Invalid direction"):
        translate_image(center_image(), "diagonal")
