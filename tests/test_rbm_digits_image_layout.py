"""Pixel-level checks for the standalone RBM sample image utility."""

import unittest

import numpy as np

from example.rbm_digits.digit_images import make_digit_strip


def sample_rows(count, size, dtype):
    """Give every sample, row and column a distinguishable pixel value."""
    return np.array(
        [[tile * 1000 + row * 10 + column
          for row in range(size) for column in range(size)]
         for tile in range(count)],
        dtype=dtype,
    )


def pixel_oracle(rows, size):
    """Place each input pixel directly, independently of reshape/stack logic."""
    expected = np.empty((size, len(rows) * size), dtype=rows.dtype)
    for tile in range(len(rows)):
        for row in range(size):
            for column in range(size):
                expected[row, tile * size + column] = rows[tile, row * size + column]
    return expected


class TestDigitStrip(unittest.TestCase):
    def assert_strip(self, supplied, rows, size):
        before = supplied.copy()
        actual = make_digit_strip(supplied, size)
        self.assertEqual(actual.shape, (size, len(rows) * size))
        self.assertEqual(actual.dtype, supplied.dtype)
        np.testing.assert_array_equal(actual, pixel_oracle(rows, size))
        np.testing.assert_array_equal(supplied, before)

    def test_sampler_batch_sizes(self):
        for count in (1, 2, 7, 20):
            with self.subTest(count=count):
                rows = sample_rows(count, 8, np.int32)
                self.assert_strip(rows, rows, 8)

    def test_strided_visible_columns(self):
        # fit slices visible columns from rows that can also contain hidden units.
        backing = np.arange(20 * 80, dtype=np.float32).reshape(20, 80)
        rows = backing[:, :64]
        self.assertFalse(rows.flags.c_contiguous)
        self.assert_strip(rows, rows, 8)

    def test_existing_reshape_compatible_inputs(self):
        rows = sample_rows(20, 8, np.float64)
        flat = np.array([rows[tile, pixel]
                         for tile in range(20) for pixel in range(64)])
        self.assert_strip(flat, rows, 8)
        small_rows = sample_rows(20, 4, np.int16)
        self.assert_strip(small_rows, small_rows, 4)


if __name__ == "__main__":
    unittest.main()
