"""Real-image collection checks; run unittest from example/qvae_mnist."""
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.helpers import get_real_images


class CountingDataLoader(DataLoader):
    def __iter__(self):
        self.iterations += 1
        for batch in super().__iter__():
            self.batches_seen += 1
            yield batch


def make_loader(rows, batch_size):
    images = torch.tensor([[4 * row + col for col in range(4)] for row in range(rows)],
                          dtype=torch.float32).reshape(rows, 4)
    loader = CountingDataLoader(TensorDataset(images, torch.arange(rows)),
                                batch_size=batch_size, shuffle=False,
                                generator=torch.Generator().manual_seed(11))
    loader.iterations = 0
    loader.batches_seen = 0
    return loader


def expected_rows(rows):
    return [[4 * row + col for col in range(4)] for row in range(rows)]


class RealImageCollectionTests(unittest.TestCase):
    def test_overshoot_preserves_prefix_without_excess_result_storage(self):
        loader = make_loader(11, 4)
        actual = get_real_images(loader, 6)
        self.assertEqual(actual.tolist(), expected_rows(6))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(actual.shape), (6, 4))
        self.assertEqual(loader.batches_seen, 2)
        self.assertEqual(actual.untyped_storage().nbytes(), 6 * 4 * 4)

    def test_requested_prefix_and_batch_boundaries(self):
        for batch_size, requested, batches in [(11, 1, 1), (4, 8, 2), (4, 10, 3)]:
            with self.subTest(batch_size=batch_size, requested=requested):
                loader = make_loader(11, batch_size)
                actual = get_real_images(loader, requested)
                self.assertEqual(actual.tolist(), expected_rows(requested))
                self.assertEqual(loader.batches_seen, batches)
                self.assertEqual(actual.untyped_storage().nbytes(), requested * 4 * 4)

    def test_short_source_and_default_return_all_available_rows(self):
        for request in (20, None):
            with self.subTest(request=request):
                loader = make_loader(11, 4)
                actual = get_real_images(loader) if request is None else get_real_images(loader, request)
                self.assertEqual(actual.tolist(), expected_rows(11))
                self.assertEqual(loader.batches_seen, 3)

    def test_numpy_integer_requests(self):
        for request in (np.int32(3), np.int64(6), np.uint64(9)):
            with self.subTest(request=request):
                loader = make_loader(11, 4)
                actual = get_real_images(loader, request)
                self.assertEqual(actual.tolist(), expected_rows(int(request)))
                self.assertEqual(loader.batches_seen, (int(request) + 3) // 4)

    def test_invalid_request_is_rejected_before_iteration(self):
        for request in (True, False, np.bool_(True), 0, -1, np.int64(0), np.int32(-2),
                        1.0, 1.5, np.float64(2), "3"):
            with self.subTest(request=request):
                loader = make_loader(11, 4)
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    get_real_images(loader, request)
                self.assertEqual(loader.iterations, 0)
                self.assertEqual(loader.batches_seen, 0)

    def test_no_rows_raise_clear_error(self):
        for source in (make_loader(0, 4), [(torch.empty((0, 4)), torch.empty(0))]):
            with self.subTest(source=type(source).__name__):
                with self.assertRaisesRegex(ValueError, "No images"):
                    get_real_images(source, 2)


if __name__ == "__main__":
    unittest.main()
