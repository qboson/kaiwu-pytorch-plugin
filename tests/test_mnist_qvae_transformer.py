"""Exercise sklearn's parameter and caching APIs for the MNIST QVAE step."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np
import pytest
import torch


class _Trainer:
    """Exercise the transformer's real data plumbing without training a QVAE."""

    train_calls = 0

    def __init__(self, config, custom_train_data, custom_test_data):
        self.config = config

    def train(self, run_tsne, compute_energy):
        type(self).train_calls += 1
        return torch.nn.Identity(), None, None

    def extract_features(self, loader, feature_type):
        features = torch.cat([data[:, :1] for data, _ in loader])
        return features, None


@pytest.fixture
def qvae_step(monkeypatch):
    pytest.importorskip("sklearn")
    example = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"

    def load_module(name, path):
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    model = load_module("mnist_transformer_test_config", example / "model/config.py")
    monkeypatch.setitem(sys.modules, "model", model)
    trainer = ModuleType("trainer")
    trainer.Trainer = _Trainer
    monkeypatch.setitem(sys.modules, "trainer", trainer)
    downstream = ModuleType("downstream")
    downstream.MLPClassifier = None
    monkeypatch.setitem(sys.modules, "downstream", downstream)
    module = load_module(
        "mnist_transformer_test_pipeline", example / "downstream/pipeline.py"
    )
    return module.PipelineTransformer, model.Config


def test_sklearn_clone_drops_training_state_but_preserves_configuration(qvae_step):
    from sklearn.base import clone

    transformer, config_type = qvae_step
    config = config_type(batch_size=4)
    step = transformer(config)
    step.trainer = object()
    step.model = torch.nn.Identity()

    copied = clone(step)

    assert copied.config is not config
    assert copied.config.batch_size == config.batch_size
    assert copied.trainer is None
    assert copied.model is None


def test_pipeline_routes_nested_configuration_to_qvae_step(qvae_step):
    from sklearn.pipeline import Pipeline

    transformer, config_type = qvae_step
    step = transformer(config_type())
    pipeline = Pipeline([("qvae", step)])
    config = config_type(batch_size=4)

    pipeline.set_params(qvae__config=config)

    assert step.config is config


def test_failed_training_leaves_the_step_unfitted(qvae_step, monkeypatch):
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    def fail_training(self, *args, **kwargs):
        raise RuntimeError("training failed")

    transformer, config_type = qvae_step
    step = transformer(config_type())
    inputs = np.arange(16, dtype=np.float32).reshape(8, 2)
    monkeypatch.setattr(_Trainer, "train", fail_training)

    with pytest.raises(RuntimeError, match="training failed"):
        step.fit(inputs)
    with pytest.raises(NotFittedError):
        check_is_fitted(step)
    with pytest.raises(ValueError, match="Model not trained"):
        step.transform(inputs)


@pytest.mark.filterwarnings("error::FutureWarning")
def test_pipeline_with_caching_fits_a_clone_of_the_qvae_step(qvae_step, monkeypatch):
    from sklearn.pipeline import Pipeline

    transformer, config_type = qvae_step
    step = transformer(config_type(batch_size=4))
    inputs = np.arange(16, dtype=np.float32).reshape(8, 2)
    monkeypatch.setattr(_Trainer, "train_calls", 0)

    with TemporaryDirectory(prefix="kpp-qvae-") as cache_dir:
        pipeline = Pipeline(
            [("qvae", step), ("end", "passthrough")], memory=cache_dir
        )
        result = pipeline.fit_transform(inputs)
        assert pipeline.named_steps["qvae"] is not step
        assert step.trainer is None
        np.testing.assert_array_equal(result, inputs[:, :1])
        np.testing.assert_array_equal(pipeline.transform(inputs), result)
        np.testing.assert_array_equal(pipeline.fit_transform(inputs), result)
        assert _Trainer.train_calls == 1
