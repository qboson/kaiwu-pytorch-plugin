"""Regression tests for FeatureExtractor feature-type validation.

``FeatureExtractor.transform`` used to fall back to the zeta branch for
any unrecognized ``feature_type`` value, silently returning the wrong
feature table while ``extract`` rejected the same configuration. Because
q and zeta share the latent dimension, the silent fallback was
undetectable from the output shape.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/qvae_mnist"))
)

import numpy as np  # noqa: E402  pylint: disable=wrong-import-position
import torch  # noqa: E402  pylint: disable=wrong-import-position
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402  pylint: disable=wrong-import-position

from model.feature_extractor import FeatureExtractor  # noqa: E402  pylint: disable=wrong-import-position


class FakeQVAE(torch.nn.Module):
    """Minimal QVAE stand-in with distinguishable q and zeta outputs."""

    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        """Return recon, posterior, q (all ones) and zeta (all zeros)."""
        batch = inputs.size(0)
        return (
            torch.zeros(batch, 8),
            None,
            torch.ones(batch, 3, device=inputs.device),
            torch.zeros(batch, 3, device=inputs.device),
        )


@pytest.fixture(name="model")
def fake_model():
    """Provide the fake QVAE on the CPU device."""
    return FakeQVAE()


@pytest.fixture(name="loader")
def toy_loader():
    """Provide a small labeled DataLoader."""
    features = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    labels = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(features, labels), batch_size=2)


def test_extract_rejects_unknown_feature_type(model, loader):
    """extract() already rejects unknown feature types."""
    extractor = FeatureExtractor(model, feature_type="logits")
    with pytest.raises(ValueError, match="feature_type must be 'q' or 'zeta'"):
        extractor.extract(loader)


def test_transform_rejects_unknown_feature_type(model):
    """transform() must reject the same values extract() rejects."""
    extractor = FeatureExtractor(model, feature_type="logits")
    with pytest.raises(ValueError, match="feature_type must be 'q' or 'zeta'"):
        extractor.transform(np.ones((4, 3), dtype=np.float32))


def test_transform_returns_requested_feature_table(model):
    """'q' and 'zeta' return distinguishable feature tables."""
    features = np.ones((4, 3), dtype=np.float32)
    q_features = FeatureExtractor(model, feature_type="q").transform(features)
    zeta_features = FeatureExtractor(model, feature_type="zeta").transform(features)
    np.testing.assert_array_equal(q_features, np.ones((4, 3)))
    np.testing.assert_array_equal(zeta_features, np.zeros((4, 3)))


def test_extract_returns_requested_feature_table(model, loader):
    """The positive path keeps working for both feature types."""
    q_features, _ = FeatureExtractor(model, feature_type="q").extract(loader)
    zeta_features, _ = FeatureExtractor(model, feature_type="zeta").extract(loader)
    np.testing.assert_array_equal(q_features.numpy(), np.ones((4, 3)))
    np.testing.assert_array_equal(zeta_features.numpy(), np.zeros((4, 3)))
