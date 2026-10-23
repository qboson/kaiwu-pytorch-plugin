"""Sampler-factory contract for the QVAE-MNIST helpers."""

import importlib.util
import os
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_sampler",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py"),
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)

import kaiwu.classical as classical  # noqa: E402


class RecordingSampler:
    """Stands in for the Kaiwu SA optimizer."""

    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingSampler.calls.append(kwargs)


def test_sa_sampler_uses_the_helper_defaults(monkeypatch):
    """The SA worker keeps alpha 0.99 and a hundred returned solutions."""
    RecordingSampler.calls = []
    monkeypatch.setattr(classical, "SimulatedAnnealingOptimizer", RecordingSampler)
    worker = helpers.init_qvae_sampler(SimpleNamespace(sampler_type="sa"))
    assert isinstance(worker, RecordingSampler)
    assert RecordingSampler.calls == [{"size_limit": 100, "alpha": 0.99}]


def test_unknown_sampler_type_is_rejected():
    """Only the two documented sampler types are accepted."""
    with pytest.raises(ValueError, match="Unsupported sampler type"):
        helpers.init_qvae_sampler(SimpleNamespace(sampler_type="bogus"))
