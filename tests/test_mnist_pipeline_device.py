"""Check device selection for the combined MNIST QVAE and MLP example."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pipeline_factory(monkeypatch):
    """Load the real factory and classifier without training dependencies."""
    pytest.importorskip("sklearn.pipeline")
    pytest.importorskip("tqdm")
    example = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
    helpers = ModuleType("utils.helpers")
    helpers.plot_training_curves = None
    monkeypatch.setitem(sys.modules, "utils.helpers", helpers)
    trainer = ModuleType("trainer")
    trainer.Trainer = None
    monkeypatch.setitem(sys.modules, "trainer", trainer)
    config = _load_module(example / "model/config.py", "mnist_pipeline_config")
    monkeypatch.setitem(sys.modules, "model", config)
    classifier = _load_module(example / "downstream/classifier.py", "mnist_pipeline_classifier")
    monkeypatch.setitem(sys.modules, "downstream", classifier)
    pipeline = _load_module(example / "downstream/pipeline.py", "mnist_pipeline_example")
    return pipeline.get_full_pipeline, config.Config


@pytest.mark.parametrize(
    ("use_cuda", "cuda_available", "kwargs", "device"),
    [
        (False, True, {}, "cpu"),
        (True, True, {}, "cuda"),
        (True, False, {}, "cpu"),
        (False, True, {"device": "cuda:1"}, "cuda:1"),
        (False, True, {"device": None}, "cuda"),
    ],
)
def test_pipeline_defaults_classifier_to_configured_device(
    pipeline_factory, monkeypatch, tmp_path, use_cuda, cuda_available, kwargs, device
):
    """Check construction without allocating a tensor on a CUDA device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch, "manual_seed", lambda _seed: None)
    get_full_pipeline, config_type = pipeline_factory
    original_kwargs = kwargs.copy()
    config = config_type(
        output_dir=str(tmp_path), use_cuda=use_cuda, classifier_kwargs=kwargs
    )

    pipeline = get_full_pipeline(config)

    assert str(pipeline.named_steps["classifier"].device) == device
    assert config.classifier_kwargs == original_kwargs
