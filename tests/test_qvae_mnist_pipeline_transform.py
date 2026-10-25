"""Contract test for ``example/qvae_mnist/downstream/pipeline.py`` transform."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "example" / "qvae_mnist" / "downstream" / "pipeline.py"


def _load_pipeline():
    names = ("model", "trainer", "downstream")
    saved = {name: sys.modules.get(name) for name in names}
    model_module = types.ModuleType("model")
    model_module.Config = object
    trainer_module = types.ModuleType("trainer")
    trainer_module.Trainer = object
    downstream_module = types.ModuleType("downstream")
    downstream_module.MLPClassifier = object
    sys.modules.update(
        {
            "model": model_module,
            "trainer": trainer_module,
            "downstream": downstream_module,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "qvae_mnist_pipeline_uut", PIPELINE
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _FakeTrainer:
    def __init__(self) -> None:
        self.seen: list[tuple[list[tuple[torch.Tensor, torch.Tensor]], str]] = []

    def extract_features(self, loader, feature_type="q"):
        batches = list(loader)
        self.seen.append((batches, feature_type))
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]]), None


def _config():
    return types.SimpleNamespace(batch_size=4, feature_type="zeta")


def test_builds_a_loader_and_delegates_to_the_trainer():
    pipeline = _load_pipeline()
    transformer = pipeline.PipelineTransformer(_config())
    transformer.trainer = _FakeTrainer()

    result = transformer.transform(np.arange(6, dtype=np.float32).reshape(3, 2))

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    batches, feature_type = transformer.trainer.seen[0]
    assert feature_type == "zeta"
    assert len(batches) == 1
    inputs, labels = batches[0]
    assert inputs.shape == (3, 2)
    assert torch.equal(inputs, torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    assert torch.equal(labels, torch.zeros(3, dtype=torch.long))


def test_uses_the_configured_batch_size():
    pipeline = _load_pipeline()
    config = _config()
    config.batch_size = 2
    transformer = pipeline.PipelineTransformer(config)
    transformer.trainer = _FakeTrainer()

    transformer.transform(np.zeros((5, 1), dtype=np.float32))

    batches, _ = transformer.trainer.seen[0]
    assert [len(inputs) for inputs, _ in batches] == [2, 2, 1]


def test_rejects_a_transformer_without_a_fitted_trainer():
    pipeline = _load_pipeline()
    transformer = pipeline.PipelineTransformer(_config())

    with pytest.raises(ValueError, match="fit"):
        transformer.transform(np.zeros((2, 1), dtype=np.float32))
