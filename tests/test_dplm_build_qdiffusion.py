"""Assembly contract for the DPLM Q-Diffusion example builder."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import pytest
import torch

import utils.dplm_builder as builder  # noqa: E402


class FakeQDiffusion:
    """Records the assembled QDiffusion keyword payload."""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeQDiffusion.last = self


@pytest.fixture()
def patched(monkeypatch):
    """Replaces weight loading and model classes with recorders."""
    calls = {"load": [], "energy": []}

    def fake_load(name, *, cfg_override=None, net_override=None, from_huggingface=True):
        calls["load"].append((name, cfg_override, net_override, from_huggingface))
        return f"backbone:{name}"

    def fake_energy(**kwargs):
        calls["energy"].append(kwargs)
        return "energy-model"

    monkeypatch.setattr(builder, "load_dplm_backbone", fake_load)
    monkeypatch.setattr(builder, "_build_energy_model", fake_energy)
    monkeypatch.setattr(
        builder, "build_dplm_token_spec", lambda model: f"spec:{model}"
    )
    monkeypatch.setattr(builder, "QDiffusion", FakeQDiffusion)
    FakeQDiffusion.last = None
    return calls


def test_explicit_settings_reach_the_models_and_the_config(patched):
    """Builder arguments are mapped onto the energy model, config and instance."""
    sampler = object()
    builder.build_qdiffusion(
        "proposal-ckpt",
        "energy-ckpt",
        bm_num_visible=7,
        bm_num_hidden=3,
        bm_sampler=sampler,
        bm_sampler_type="cim",
        bm_sampler_kwargs={"tmp_dir": "checkpoints"},
        num_candidates=3,
        proposal_temperature=0.1,
        proposal_noise_scale=2.0,
        energy_temperature=0.5,
        disable_resample=True,
        resample_ratio=0.5,
        resample_top_p=0.8,
        freeze_proposal=False,
        dtype=torch.float64,
        device="cpu",
    )
    assert patched["load"] == [
        ("proposal-ckpt", None, None, True),
        ("energy-ckpt", None, None, True),
    ]
    energy = patched["energy"][0]
    assert energy["energy_backbone"] == "backbone:energy-ckpt"
    assert energy["bm_num_visible"] == 7
    assert energy["bm_num_hidden"] == 3
    assert energy["bm_sampler"] is sampler
    assert energy["bm_sampler_type"] == "cim"
    assert energy["bm_sampler_kwargs"] == {"tmp_dir": "checkpoints"}

    instance = FakeQDiffusion.last
    assert instance.kwargs["proposal_model"] == "backbone:proposal-ckpt"
    assert instance.kwargs["energy_model"] == "energy-model"
    assert instance.kwargs["token_spec"] == "spec:backbone:proposal-ckpt"
    assert instance.kwargs["dtype"] is torch.float64
    assert instance.kwargs["device"] == "cpu"
    assert instance.kwargs["freeze_proposal"] is False
    config = instance.kwargs["config"]
    assert config.num_candidates == 3
    assert config.proposal_temperature == pytest.approx(0.1)
    assert config.proposal_noise_scale == pytest.approx(2.0)
    assert config.energy_temperature == pytest.approx(0.5)
    assert config.disable_resample is True
    assert config.resample_ratio == pytest.approx(0.5)
    assert config.resample_top_p == pytest.approx(0.8)


def test_defaults_match_the_documented_example_settings(patched):
    """The builder defaults stay at 256/128 visible/hidden and SA sampling."""
    builder.build_qdiffusion("p", "e")
    energy = patched["energy"][0]
    assert energy["bm_num_visible"] == 256
    assert energy["bm_num_hidden"] == 128
    assert energy["bm_sampler"] is None
    assert energy["bm_sampler_type"] == "sa"
    assert energy["bm_sampler_kwargs"] is None
    instance = FakeQDiffusion.last
    assert instance.kwargs["dtype"] is torch.float32
    assert instance.kwargs["device"] is None
    assert instance.kwargs["freeze_proposal"] is True
    config = instance.kwargs["config"]
    assert config.num_candidates == 1
    assert config.proposal_temperature == pytest.approx(0.0)
    assert config.energy_temperature == pytest.approx(1.0)
    assert config.resample_ratio == pytest.approx(0.25)
    assert config.resample_top_p == pytest.approx(0.95)


def test_backward_compatible_alias_forwards_arguments(patched, monkeypatch):
    """build_dplm_qdiffusion keeps calling the new builder unchanged."""
    recorded = {}

    def fake_builder(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return "built"

    monkeypatch.setattr(builder, "build_qdiffusion", fake_builder)
    assert builder.build_dplm_qdiffusion("p", "e", num_candidates=2) == "built"
    assert recorded == {"args": ("p", "e"), "kwargs": {"num_candidates": 2}}
