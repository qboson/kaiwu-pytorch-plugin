# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the RBMRunner.plot_images image layout.

``plot_images`` hard-coded ``view(8, 8)``, which only fits the bundled 8x8
digits dataset.  Any other feature width crashed inside ``Tensor.view``.
The layout is now derived from the data width, and non-square widths raise a
clear ``ValueError``, matching the sibling ``plot_weights`` fix.
"""

import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")

# The backend must be selected before pyplot is imported.
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../example/rbm_digits")
    ),
)

# rbm_digits lives under example/, which is added to sys.path above.
# pylint: disable=wrong-import-position,import-error
from rbm_digits import RBMRunner  # noqa: E402


def _make_runner(num_visible):
    """Build an RBMRunner whose RBM has ``num_visible`` visible units."""
    runner = RBMRunner.__new__(RBMRunner)
    runner.rbm = type("RBM", (), {})()
    runner.rbm.num_visible = num_visible
    runner.rbm.num_hidden = 4
    runner.device = torch.device("cpu")
    runner.sampler = None
    return runner


class TestPlotImagesLayout(unittest.TestCase):
    """The image layout must follow the data width, not the bundled dataset."""

    def _capture_layout(self, runner, num_visible, num_samples=2):
        """Run plot_images and return the (side, side) shapes actually drawn."""
        images = torch.zeros(num_samples, num_visible)
        labels = np.zeros(num_samples)

        seen = []
        original_imshow = plt.Axes.imshow

        def spy(self, data, *args, **kwargs):
            seen.append(tuple(np.asarray(data).shape))
            return original_imshow(self, data, *args, **kwargs)

        plt.Axes.imshow = spy
        try:
            # Reconstruct so plot_images has a tensor to draw.
            runner.rbm.sample = lambda sampler: torch.zeros(
                num_samples, num_visible + runner.rbm.num_hidden
            )
            runner.rbm.get_hidden = lambda x: torch.zeros(
                num_samples, runner.rbm.num_hidden
            )
            with torch.no_grad():
                runner.plot_images(images, labels)
        finally:
            plt.Axes.imshow = original_imshow
            plt.close("all")
        return seen

    def test_bundled_sixty_four_visible_still_renders_eight_by_eight(self):
        """The bundled digits width is unchanged: 64 -> 8x8."""
        runner = _make_runner(64)
        shapes = self._capture_layout(runner, 64)
        self.assertTrue(shapes)
        for shape in shapes:
            self.assertEqual(shape, (8, 8))

    def test_other_square_widths_render(self):
        """Square widths other than 64 now render instead of crashing."""
        for width, side in ((16, 4), (256, 16)):
            with self.subTest(width=width):
                runner = _make_runner(width)
                shapes = self._capture_layout(runner, width)
                self.assertTrue(shapes)
                for shape in shapes:
                    self.assertEqual(shape, (side, side))

    def test_non_square_width_raises_value_error(self):
        """Non-square widths get an actionable error, not a Tensor.view error."""
        for width in (12, 128):
            with self.subTest(width=width):
                runner = _make_runner(width)
                with self.assertRaises(ValueError) as ctx:
                    self._capture_layout(runner, width)
                self.assertIn("square image", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
