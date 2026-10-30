"""FeatureExtractor must validate feature_type up front."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_PATH = (
    REPO_ROOT / "example" / "qvae_mnist" / "model" / "feature_extractor.py"
)


def _load_extractor_class():
    spec = importlib.util.spec_from_file_location(
        "qvae_mnist_feature_extractor", EXTRACTOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeatureExtractor


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        zeros = torch.zeros(x.size(0), 3)
        return zeros, None, x, x + self.scale


def test_unknown_feature_type_is_rejected_at_construction() -> None:
    extractor_cls = _load_extractor_class()
    with pytest.raises(ValueError, match="feature_type must be 'q' or 'zeta'"):
        extractor_cls(_Model(), feature_type="bogus")


def test_valid_feature_types_still_extract() -> None:
    extractor_cls = _load_extractor_class()
    inputs = np.zeros((2, 4), dtype=np.float32)

    q_extractor = extractor_cls(_Model(), feature_type="q")
    np.testing.assert_allclose(q_extractor.transform(inputs), np.zeros((2, 4)))

    z_extractor = extractor_cls(_Model(), feature_type="zeta")
    np.testing.assert_allclose(z_extractor.transform(inputs), np.zeros((2, 4)))
