"""Reject unknown dataset names before the real loader accesses data."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


class DatasetNameTests(unittest.TestCase):
    """Exercise the normal loader entry without constructing a dataset."""

    @classmethod
    def setUpClass(cls):
        try:
            from example.qvae_mnist.utils.loadMNIST import loadMNIST
        except ModuleNotFoundError as exc:
            if exc.name == "torchvision" and importlib.util.find_spec("torchvision") is None:
                raise unittest.SkipTest("Optional torchvision is not installed") from exc
            raise
        cls.load_mnist = staticmethod(loadMNIST)

    def check_unknown_name(self, name):
        with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data"
            data_path.mkdir()
            try:
                try:
                    self.load_mnist(name=name, data_path=str(data_path))
                except Exception as exc:
                    self.assertIsInstance(exc, ValueError)
                    self.assertEqual(
                        str(exc),
                        f"Unsupported dataset {name!r}. Choose from: fashion-mnist, kmnist, mnist",
                    )
                else:
                    self.fail("Loader accepted an unsupported dataset name")
            finally:
                self.assertEqual(list(data_path.iterdir()), [])

    def test_unknown_name(self):
        self.check_unknown_name("not-a-dataset")

    def test_typo(self):
        self.check_unknown_name("mnist2")

    def test_case_sensitive_name(self):
        self.check_unknown_name("MNIST")


if __name__ == "__main__":
    unittest.main()
