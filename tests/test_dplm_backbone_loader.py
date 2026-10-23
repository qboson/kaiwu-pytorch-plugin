"""Backbone-loading contract for the DPLM example factory."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import pytest

import utils.dplm_builder as builder  # noqa: E402


class RecordingBackbone:
    """Sentinel returned by the patched pretrained loader."""


@pytest.fixture()
def recording_loader(monkeypatch):
    """Records ``from_pretrained`` calls instead of loading weights."""
    calls = []

    def fake_from_pretrained(cls, name, **kwargs):
        calls.append((name, kwargs))
        return RecordingBackbone()

    monkeypatch.setattr(
        builder.DPLMBackbone,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    return calls


def test_overrides_and_source_flag_are_forwarded(recording_loader):
    """The checkpoint name and all three loader options reach the wrapper."""
    result = builder.load_dplm_backbone(
        "airkingbd/dplm_150m",
        cfg_override={"hidden_size": 8},
        net_override={"num_hidden_layers": 1},
        from_huggingface=False,
    )
    assert isinstance(result, RecordingBackbone)
    assert recording_loader == [
        (
            "airkingbd/dplm_150m",
            {
                "cfg_override": {"hidden_size": 8},
                "net_override": {"num_hidden_layers": 1},
                "from_huggingface": False,
            },
        )
    ]


def test_defaults_load_from_huggingface_without_overrides(recording_loader):
    """Omitted overrides stay None and the HF source stays enabled."""
    builder.load_dplm_backbone("local-checkpoint")
    assert recording_loader == [
        (
            "local-checkpoint",
            {"cfg_override": None, "net_override": None, "from_huggingface": True},
        )
    ]
