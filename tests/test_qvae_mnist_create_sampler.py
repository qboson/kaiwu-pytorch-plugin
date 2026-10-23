"""Sampler-selection contract for the MNIST QVAE."""

import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

import model.model as model_module  # noqa: E402
from model.model import MnistQVAE  # noqa: E402


class RecordingSampler:
    """Stands in for the Kaiwu SA sampler and records its kwargs."""

    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingSampler.calls.append(kwargs)


def build_model(sampler_type="sa"):
    """Builds only the attributes ``_create_sampler`` reads."""
    model = MnistQVAE.__new__(MnistQVAE)
    model.config = SimpleNamespace(sampler_type=sampler_type)
    return model


@pytest.fixture()
def recording_sa(monkeypatch):
    """Replaces the SA class for one test."""
    RecordingSampler.calls = []
    monkeypatch.setattr(model_module, "SimulatedAnnealingOptimizer", RecordingSampler)
    return RecordingSampler


def test_sa_is_built_with_the_example_alpha(recording_sa):
    """The default SA sampler keeps alpha 0.95."""
    sampler = build_model("sa")._create_sampler()
    assert isinstance(sampler, RecordingSampler)
    assert recording_sa.calls == [{"alpha": 0.95}]


def test_explicit_argument_wins_over_the_config(recording_sa):
    """The method argument overrides ``config.sampler_type``."""
    sampler = build_model("bogus")._create_sampler("sa")
    assert isinstance(sampler, RecordingSampler)
    assert recording_sa.calls == [{"alpha": 0.95}]


def test_unknown_sampler_type_is_rejected(recording_sa):
    """Only ``sa`` and ``cim`` are supported."""
    with pytest.raises(ValueError, match="Unsupported sampler type"):
        build_model("bogus")._create_sampler()
