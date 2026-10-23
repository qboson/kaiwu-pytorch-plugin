"""Network-construction contract for the DPLM example backbone."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import models.backbone as backbone  # noqa: E402


class FakeNet:
    """Records construction and state-dict loading."""

    instances = []

    def __init__(self, config, dropout=None):
        self.config = config
        self.dropout = dropout
        self.loaded = []
        FakeNet.instances.append(self)

    def load_state_dict(self, state, strict=True):
        self.loaded.append((dict(state), strict))


class FakeAutoConfig:
    """Returns one sentinel config object."""

    calls = []

    @classmethod
    def from_pretrained(cls, name):
        cls.calls.append(name)
        return "CONFIG"


class FakePretrainedModel:
    """Pretrained model stub exposing one state dict."""

    calls = []

    @classmethod
    def from_pretrained(cls, name):
        cls.calls.append(name)
        return SimpleNamespace(state_dict=lambda: {"weight": 1.0})


@pytest.fixture()
def patched(monkeypatch):
    """Replaces the transformers and ESM class lookups."""
    FakeNet.instances = []
    FakeAutoConfig.calls = []
    FakePretrainedModel.calls = []
    monkeypatch.setattr(backbone, "AutoConfig", FakeAutoConfig)
    monkeypatch.setattr(backbone, "_EsmForDPLM", FakeNet)
    monkeypatch.setattr(backbone, "AutoModelForMaskedLM", FakePretrainedModel)


def make_cfg(**net_overrides):
    """Builds the minimal config attribute tree ``get_net`` reads."""
    net = SimpleNamespace(
        arch_type="esm",
        name="fake-arch",
        dropout=0.1,
        pretrain=False,
        pretrained_model_name_or_path="unused",
    )
    for key, value in net_overrides.items():
        setattr(net, key, value)
    return SimpleNamespace(net=net)


def test_unsupported_architecture_is_not_implemented(patched):
    """Only the patched ESM family is wired in the example."""
    with pytest.raises(NotImplementedError, match="Unsupported arch_type"):
        backbone.get_net(make_cfg(arch_type="linear"))


def test_untrained_network_is_built_from_the_auto_config(patched):
    """``pretrain=False`` builds the net and loads no weights."""
    net = backbone.get_net(make_cfg())
    assert isinstance(net, FakeNet)
    assert net.config == "CONFIG"
    assert net.dropout == 0.1
    assert net.loaded == []
    assert FakeAutoConfig.calls == ["fake-arch"]
    assert FakePretrainedModel.calls == []


def test_local_checkpoint_prefix_is_stripped_before_loading(patched, tmp_path, monkeypatch):
    """Local DPLM artifacts keep a ten-character prefix that must be removed."""
    checkpoint = tmp_path / "artifact.pt"
    checkpoint.write_bytes(b"stub")
    calls = []

    def fake_load(path, map_location="cpu"):
        calls.append((str(path), map_location))
        return {"state_dict": {"0123456789abc": 5, "0123456789def": 6}}

    monkeypatch.setattr(backbone.torch, "load", fake_load)
    net = backbone.get_net(
        make_cfg(pretrain=True, pretrained_model_name_or_path=str(checkpoint))
    )
    assert calls == [(str(checkpoint), "cpu")]
    assert net.loaded == [({"abc": 5, "def": 6}, True)]


def test_hub_checkpoint_loads_the_pretrained_state_dict(patched):
    """A non-local path loads through the transformers masked-LM loader."""
    backbone.get_net(make_cfg(pretrain=True, pretrained_model_name_or_path="org/model"))
    assert FakePretrainedModel.calls == ["org/model"]
    assert FakeNet.instances[0].loaded == [({"weight": 1.0}, True)]
