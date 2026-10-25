"""Data-preparation contract for the QVAE-MNIST trainer."""

import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_PATH = list(sys.path)
sys.path[:] = [p for p in ORIGINAL_PATH if not p.endswith("qdiffusion") and not p.endswith("dplm")]
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]
for _name in [m for m in list(sys.modules) if m == "trainer" or m.startswith("trainer.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

trainer_module = importlib.import_module("trainer.trainer")
sys.path[:] = ORIGINAL_PATH
Trainer = trainer_module.Trainer


def test_custom_data_is_wrapped_and_the_global_mean_is_computed():
    """Custom arrays become loaders and the mean is weighted by sample count."""
    x_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    y_train = np.array([0, 1, 0])
    x_test = np.array([[7.0, 8.0]], dtype=np.float32)
    y_test = np.array([1])

    trainer = Trainer.__new__(Trainer)
    trainer.custom_train_data = (x_train, y_train)
    trainer.custom_test_data = (x_test, y_test)
    trainer.config = SimpleNamespace(batch_size=2)
    train_loader, test_loader = trainer._setup_data()

    batches = list(train_loader)
    assert [len(batch[1]) for batch in batches] == [2, 1]
    assert list(test_loader)[0][0].shape == (1, 2)
    expected_mean = torch.tensor(x_train).mean()
    assert torch.isclose(trainer.dataset_mean, expected_mean)
