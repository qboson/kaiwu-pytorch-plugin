# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
# Modified: infer the image count instead of requiring twenty samples.
"""NumPy image layout shared by the RBM example and standalone array callers."""

import numpy as np


def make_digit_strip(X, size=8):
    """Join flattened square images into a horizontal strip.

    Args:
        X: NumPy array containing images of ``size * size`` pixels. The
            image count is inferred from the total number of pixels.
        size: Side length of each square image.

    Returns:
        A NumPy array of shape ``(size, image_count * size)`` with the input dtype.
    """
    digits = X.reshape(-1, size, size)
    image = np.hstack(digits)
    return image
