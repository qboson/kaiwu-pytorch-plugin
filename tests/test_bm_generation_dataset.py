"""Behavioral coverage for the BM example's real CSV row dataset."""
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from example.bm_generation.data_loader import CSVDataset


class TestCSVDataset(unittest.TestCase):
    def test_list_rows_are_float32_tensors(self):
        dataset = CSVDataset([[1, -2], [3, 4], [5, 6]])
        self.assertEqual(len(dataset), 3)
        for index, expected in enumerate(([1.0, -2.0], [3.0, 4.0], [5.0, 6.0])):
            with self.subTest(index=index):
                row = dataset[index]
                self.assertIsInstance(row, torch.Tensor)
                self.assertEqual(row.dtype, torch.float32)
                self.assertEqual(tuple(row.shape), (2,))
                self.assertEqual(row.tolist(), expected)

    def test_numpy_float64_rows_convert_to_float32(self):
        source = np.array([[0.25, -1.5], [2.75, 8.0]], dtype=np.float64)
        dataset = CSVDataset(source)
        self.assertEqual(len(dataset), 2)
        for index, expected in enumerate(([0.25, -1.5], [2.75, 8.0])):
            with self.subTest(index=index):
                row = dataset[index]
                self.assertIsInstance(row, torch.Tensor)
                self.assertEqual(row.dtype, torch.float32)
                self.assertEqual(tuple(row.shape), (2,))
                self.assertEqual(row.tolist(), expected)
        self.assertEqual(source.dtype, np.float64)

    def test_returned_rows_do_not_alias_list_or_numpy_sources(self):
        sources = [
            [[1.0, 2.0], [3.0, 4.0]],
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        ]
        for source in sources:
            with self.subTest(source_type=type(source).__name__):
                dataset = CSVDataset(source)
                first = dataset[0]
                second = dataset[0]
                first[0] = 99.0
                self.assertEqual(float(source[0][0]), 1.0)
                self.assertEqual(second.tolist(), [1.0, 2.0])
                self.assertEqual(dataset[0].tolist(), [1.0, 2.0])
                source[0][1] = -7.0
                self.assertEqual(second.tolist(), [1.0, 2.0])

    def test_negative_indices_and_out_of_range_errors(self):
        sources = [
            [[1, 2], [3, 4], [5, 6]],
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
        ]
        for source in sources:
            with self.subTest(source_type=type(source).__name__):
                dataset = CSVDataset(source)
                self.assertEqual(dataset[-1].tolist(), [5.0, 6.0])
                self.assertEqual(dataset[-len(dataset)].tolist(), [1.0, 2.0])
                for index in (len(dataset), -len(dataset) - 1):
                    with self.subTest(index=index):
                        with self.assertRaises(IndexError):
                            dataset[index]

    def test_default_collation_preserves_rows_dtype_and_partial_batch(self):
        sources = [
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64),
        ]
        for source in sources:
            with self.subTest(source_type=type(source).__name__):
                loader = DataLoader(CSVDataset(source), batch_size=3, shuffle=False)
                batches = list(loader)
                self.assertEqual(len(batches), 2)
                for batch, expected in zip(
                    batches,
                    ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0]]),
                ):
                    self.assertIsInstance(batch, torch.Tensor)
                    self.assertEqual(batch.dtype, torch.float32)
                    self.assertEqual(tuple(batch.shape), (len(expected), 2))
                    self.assertEqual(batch.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
