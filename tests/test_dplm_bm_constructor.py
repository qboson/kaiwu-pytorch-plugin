"""Constructor contract for the DPLM conditioned BM reranker."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import models.bm as bm  # noqa: E402


class FakeEncoder:
    """Minimal encoder exposing the hidden size the projector uses."""

    hidden_size = 5


class DummySampler:
    """Marker object; the constructor never samples."""


def test_projector_width_matches_two_encoded_sequences(monkeypatch):
    """The projector consumes noisy+candidate features and emits BM-visible width."""
    built = []

    def fake_build(**kwargs):
        built.append(kwargs)
        return "BUILT"

    monkeypatch.setattr(bm, "build_bm_sampler", fake_build)
    model = bm.BMConditionedEnergyModel(
        encoder=FakeEncoder(),
        bm_num_visible=7,
        bm_num_hidden=3,
        sampler=None,
        sampler_type="cim",
        sampler_kwargs={"tmp_dir": "checkpoints"},
    )
    assert model.feature_projector.in_features == 10
    assert model.feature_projector.out_features == 7
    assert model.sampler_type == "cim"
    assert model.sampler_kwargs == {"tmp_dir": "checkpoints"}
    assert built == [{"sampler_type": "cim", "sampler_kwargs": {"tmp_dir": "checkpoints"}}]
    assert model.sampler == "BUILT"


def test_explicit_sampler_bypasses_the_sampler_factory(monkeypatch):
    """A supplied sampler is used as-is without building another one."""
    built = []
    monkeypatch.setattr(bm, "build_bm_sampler", lambda **kwargs: built.append(kwargs))
    sampler = DummySampler()
    model = bm.BMConditionedEnergyModel(
        encoder=FakeEncoder(),
        bm_num_visible=4,
        bm_num_hidden=2,
        sampler=sampler,
    )
    assert built == []
    assert model.sampler is sampler
    assert model.sampler_type == "sa"
    assert model.sampler_kwargs == {}


def test_sampler_kwargs_are_copied_from_the_caller(monkeypatch):
    """Later mutation of the caller dictionary cannot change the model settings."""
    monkeypatch.setattr(bm, "build_bm_sampler", lambda **kwargs: "BUILT")
    kwargs = {"tmp_dir": "a"}
    model = bm.BMConditionedEnergyModel(
        encoder=FakeEncoder(),
        bm_num_visible=4,
        bm_num_hidden=2,
        sampler=DummySampler(),
        sampler_kwargs=kwargs,
    )
    kwargs["tmp_dir"] = "b"
    kwargs["wait"] = True
    assert model.sampler_kwargs == {"tmp_dir": "a"}


def test_energy_bm_has_the_visible_plus_hidden_width():
    """The underlying BM exposes one node per visible and hidden unit."""
    model = bm.BMConditionedEnergyModel(
        encoder=FakeEncoder(),
        bm_num_visible=4,
        bm_num_hidden=2,
        sampler=DummySampler(),
    )
    assert model.energy_bm.num_nodes == 6
    assert isinstance(model.feature_projector, torch.nn.Linear)
