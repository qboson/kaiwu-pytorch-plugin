"""Regression test for ``PipelineTransformer.fit`` on small subsets.

``run_pipeline.py --num-train-samples N`` builds a random subset of the
dataset; for small N some classes can have a single member.  The transformer
always passed ``stratify=y`` to ``train_test_split``, which rejects that:

    ValueError: The least populated class in y has only 1 member, which is too
    few. The minimum number of groups for any class cannot be less than 2.
"""

import os
import sys
import types

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import downstream.pipeline as pipeline_module  # noqa: E402
from downstream.pipeline import PipelineTransformer  # noqa: E402


class _StubTrainer:
    """Captures the data split without training (training needs a licence)."""

    def __init__(self, config=None, custom_train_data=None, custom_test_data=None):
        self.config = config
        self.train_data = custom_train_data
        self.test_data = custom_test_data

    def train(self, run_tsne=False, compute_energy=False):
        return "model", [], []


@pytest.fixture()
def _stub_trainer(monkeypatch):
    monkeypatch.setattr(pipeline_module, "Trainer", _StubTrainer)


def _config(**kwargs):
    return types.SimpleNamespace(run_tsne=False, compute_energy=False, **kwargs)


def test_singleton_class_falls_back_to_random_split(_stub_trainer):
    rng = np.random.RandomState(0)
    x = rng.rand(30, 784).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10 + [2] * 9 + [3])  # class 3 has 1 member

    transformer = PipelineTransformer(_config())
    transformer.fit(x, y)  # must not raise

    train_x, _ = transformer.trainer.train_data
    val_x, _ = transformer.trainer.test_data
    assert train_x.shape[0] + val_x.shape[0] == x.shape[0]


def test_stratified_split_is_kept_when_possible(_stub_trainer):
    rng = np.random.RandomState(0)
    x = rng.rand(40, 784).astype(np.float32)
    y = np.array([0, 1] * 20)

    transformer = PipelineTransformer(_config())
    transformer.fit(x, y)

    _, train_y = transformer.trainer.train_data
    _, val_y = transformer.trainer.test_data
    # Both classes present in both splits => stratification active
    assert set(np.unique(train_y)) == {0, 1}
    assert set(np.unique(val_y)) == {0, 1}
