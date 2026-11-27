"""MLPClassifier must satisfy the sklearn estimator parameter contract."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "example" / "qvae_mnist"))

from downstream.classifier import MLPClassifier  # noqa: E402  pylint: disable=wrong-import-position


def test_get_params_exposes_constructor_arguments() -> None:
    classifier = MLPClassifier(lr_mlp=0.01, batch_size_mlp=32, epochs_mlp=7)
    params = classifier.get_params()

    assert params["lr_mlp"] == 0.01
    assert params["batch_size_mlp"] == 32
    assert params["epochs_mlp"] == 7


def test_set_params_updates_the_values_used_by_fit() -> None:
    classifier = MLPClassifier(lr_mlp=0.01, batch_size_mlp=32, epochs_mlp=7)
    classifier.set_params(lr_mlp=0.5, batch_size_mlp=8, epochs_mlp=2)

    assert classifier.lr == 0.5
    assert classifier.batch_size == 8
    assert classifier.epochs == 2
