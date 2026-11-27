"""Check subset-count validation before the real loader constructs a dataset."""
from enum import IntEnum
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class DatasetBoundaryReached(RuntimeError):
    """Stop at the dataset-construction boundary without returning a dataset."""


class IntegralCount(IntEnum):
    THREE = 3


class MNISTCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.loader = importlib.import_module("example.qvae_mnist.utils.loadMNIST")
        except ModuleNotFoundError as exc:
            if exc.name == "torchvision" and importlib.util.find_spec("torchvision") is None:
                raise unittest.SkipTest("Optional torchvision is not installed") from exc
            raise
        import numpy as np
        cls.numpy = np

    def test_invalid_counts_before_dataset_construction(self):
        for parameter in ("num_evts_train", "num_evts_test"):
            for value in (-1, 0, 1.5, 2.0, False):
                with self.subTest(kind="invalid", parameter=parameter, value=repr(value)):
                    with tempfile.TemporaryDirectory() as directory:
                        with patch.object(self.loader, "MNIST", side_effect=DatasetBoundaryReached("MNIST construction reached")) as constructor:
                            try:
                                try:
                                    self.loader.loadMNIST(data_path=directory, **{parameter: value})
                                except Exception as exc:
                                    self.assertIsInstance(exc, ValueError)
                                    self.assertIn(parameter, str(exc))
                                    self.assertIn("positive integer", str(exc))
                                else:
                                    self.fail("Loader accepted an invalid count without an error")
                                constructor.assert_not_called()
                            finally:
                                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_positive_integer_counts_reach_dataset_boundary(self):
        values = (1, self.numpy.int64(3), IntegralCount.THREE, True, 100000)
        for value in values:
            with self.subTest(kind="positive_boundary", value=repr(value), type=type(value).__name__):
                with tempfile.TemporaryDirectory() as directory:
                    with patch.object(self.loader, "MNIST", side_effect=DatasetBoundaryReached("MNIST construction reached")) as constructor:
                        try:
                            with self.assertRaises(DatasetBoundaryReached):
                                self.loader.loadMNIST(data_path=directory, num_evts_train=value, num_evts_test=value)
                            constructor.assert_called_once()
                        finally:
                            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unknown_dataset_name_retains_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.loader, "MNIST", side_effect=DatasetBoundaryReached("MNIST construction reached")) as constructor:
                try:
                    with self.assertRaises(ValueError) as caught:
                        self.loader.loadMNIST(name="mnist2", data_path=directory, num_evts_train=0, num_evts_test=0)
                    self.assertEqual(
                        str(caught.exception),
                        "Unsupported dataset 'mnist2'. Choose from: fashion-mnist, kmnist, mnist",
                    )
                    constructor.assert_not_called()
                finally:
                    self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
