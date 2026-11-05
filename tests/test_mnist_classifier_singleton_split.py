# -*- coding: utf-8 -*-
"""The MNIST downstream MLP must train when a class has one sample."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

QVAE_MNIST_ROOT = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
if str(QVAE_MNIST_ROOT) not in sys.path:
    sys.path.insert(0, str(QVAE_MNIST_ROOT))

_CLASSIFIER_PATH = QVAE_MNIST_ROOT / "downstream" / "classifier.py"
_SPEC = importlib.util.spec_from_file_location("classifier_under_test", _CLASSIFIER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
classifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = classifier
_SPEC.loader.exec_module(classifier)


def test_singleton_class_falls_back_to_unstratified_split(monkeypatch) -> None:
    """stratify=y rejects a class with one member before any training starts."""
    monkeypatch.setattr(classifier, "plot_training_curves", lambda **kwargs: None)
    model = classifier.MLPClassifier(
        input_dim=4,
        hidden_dims=[4],
        output_dim=3,
        epochs_mlp=1,
        batch_size_mlp=6,
        random_state=0,
        device="cpu",
    )
    monkeypatch.setattr(
        model,
        "_eval_mlp_epoch",
        lambda **kwargs: (100.0, 0.0),
    )

    features = np.arange(24, dtype=np.float32).reshape(6, 4)
    labels = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)

    fitted = model.fit(features, labels)

    assert fitted is model
    assert model.model is not None
    assert set(model.classes_.tolist()) == {0, 1, 2}
