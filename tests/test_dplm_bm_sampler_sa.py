"""SA-builder contract for the DPLM BM sampler helper."""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import kaiwu.classical as classical  # noqa: E402
from models.sampler import build_bm_sampler  # noqa: E402


class RecordingOptimizer:
    """Stands in for the Kaiwu SA optimizer and records its kwargs."""

    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingOptimizer.calls.append(kwargs)


@pytest.fixture()
def recording_sa(monkeypatch):
    """Replaces the SA class only for the duration of one test."""
    original = classical.SimulatedAnnealingOptimizer
    RecordingOptimizer.calls = []
    monkeypatch.setattr(classical, "SimulatedAnnealingOptimizer", RecordingOptimizer)
    yield RecordingOptimizer
    monkeypatch.setattr(classical, "SimulatedAnnealingOptimizer", original)


def test_sa_defaults_match_the_self_contained_example_settings(recording_sa):
    """The local SA path keeps alpha 0.95 and ten returned solutions."""
    sampler = build_bm_sampler(sampler_type="sa")
    assert isinstance(sampler, RecordingOptimizer)
    assert recording_sa.calls == [{"alpha": 0.95, "size_limit": 10}]


def test_sa_kwargs_override_defaults_without_mutating_the_caller(recording_sa):
    """Caller kwargs win, keep the missing default, and are not modified."""
    kwargs = {"alpha": 0.5, "iterations_per_t": 3}
    build_bm_sampler(sampler_type="sa", sampler_kwargs=kwargs)
    assert recording_sa.calls == [
        {"alpha": 0.5, "size_limit": 10, "iterations_per_t": 3}
    ]
    assert kwargs == {"alpha": 0.5, "iterations_per_t": 3}


def test_unknown_sampler_type_is_rejected(recording_sa):
    """An unknown backend fails with the supported families named."""
    with pytest.raises(ValueError, match="Unsupported BM sampler_type"):
        build_bm_sampler(sampler_type="bogus")
