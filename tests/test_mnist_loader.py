# -*- coding: utf-8 -*-
"""Validation tests for the qvae_mnist MNIST loading helper."""

import importlib.util
import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader, Subset  # noqa: F401  (mirrors helper imports)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER_PATH = os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "loadMNIST.py")


def _load_helper():
    spec = importlib.util.spec_from_file_location("kpp_load_mnist_under_test", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeDataset:
    """Minimal stand-in for a torchvision MNIST-style dataset."""

    def __init__(self, root, train, download, transform):  # noqa: A002
        self.train = train
        self.transform = transform
        self._items = list(range(8))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index], 0


class TestLoadMNISTCountValidation(unittest.TestCase):
    """Invalid subset counts must fail before any dataset is constructed."""

    def setUp(self):
        self.helper = _load_helper()

    def test_negative_train_count_rejected(self):
        with self.assertRaises(ValueError):
            self.helper.loadMNIST(num_evts_train=-1, num_evts_test=10)

    def test_zero_test_count_rejected(self):
        with self.assertRaises(ValueError):
            self.helper.loadMNIST(num_evts_train=100, num_evts_test=0)

    def test_float_count_rejected(self):
        with self.assertRaises(ValueError):
            self.helper.loadMNIST(num_evts_train=60000.5, num_evts_test=100)

    def test_valid_counts_subset_dataset(self):
        self.helper.MNIST = _FakeDataset
        train_loader, test_loader = self.helper.loadMNIST(
            num_evts_train=4, num_evts_test=3, batch_size=2
        )
        self.assertEqual(len(train_loader.dataset), 4)
        self.assertEqual(len(test_loader.dataset), 3)

    def test_counts_larger_than_dataset_keep_everything(self):
        self.helper.MNIST = _FakeDataset
        train_loader, test_loader = self.helper.loadMNIST(
            num_evts_train=100, num_evts_test=100, batch_size=2
        )
        self.assertEqual(len(train_loader.dataset), 8)
        self.assertEqual(len(test_loader.dataset), 8)


if __name__ == "__main__":
    unittest.main()
