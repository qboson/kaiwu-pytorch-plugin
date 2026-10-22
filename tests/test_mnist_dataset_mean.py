"""Regression test for the MNIST QVAE per-pixel dataset mean."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("gif")
pytest.importorskip("imageio")
pytest.importorskip("torchvision")
pytest.importorskip("torchmetrics")
pytest.importorskip("pandas")
pytest.importorskip("seaborn")
pytest.importorskip("tqdm")

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/qvae_mnist"))
)
from model.config import Config  # noqa: E402  pylint: disable=wrong-import-position
from trainer.trainer import Trainer  # noqa: E402  pylint: disable=wrong-import-position


def _feature_matrix(num_samples=8):
    features = np.zeros((num_samples, 4), dtype=np.float32)
    features[:, 0] = 0.0
    features[:, 1] = 0.25
    features[:, 2] = 0.5
    features[:, 3] = 1.0
    return features


def test_dataset_mean_is_computed_per_feature(tmp_path):
    features = _feature_matrix()
    labels = np.zeros(features.shape[0], dtype=np.int64)
    config = Config(
        "QVAE",
        output_dir=str(tmp_path),
        batch_size=4,
        num_train_samples=features.shape[0],
        num_test_samples=features.shape[0],
    )
    trainer = Trainer(
        config,
        custom_train_data=(features, labels),
        custom_test_data=(features, labels),
    )

    trainer._setup_data()  # pylint: disable=protected-access

    assert trainer.dataset_mean.shape == (features.shape[1],)
    assert np.allclose(trainer.dataset_mean.numpy(), [0.0, 0.25, 0.5, 1.0])
