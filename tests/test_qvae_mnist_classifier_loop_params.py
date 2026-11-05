"""MLPClassifier must validate its training-loop sizes."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "example" / "qvae_mnist"))

from downstream.classifier import MLPClassifier  # noqa: E402  pylint: disable=wrong-import-position


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs_mlp": 0},
        {"epochs_mlp": -3},
        {"batch_size_mlp": 0},
        {"batch_size_mlp": -8},
    ],
)
def test_invalid_loop_parameters_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        MLPClassifier(**kwargs)


def test_valid_parameters_still_construct() -> None:
    classifier = MLPClassifier(epochs_mlp=3, batch_size_mlp=16)
    assert classifier.epochs == 3
    assert classifier.batch_size == 16
