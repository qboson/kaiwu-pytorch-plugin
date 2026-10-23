"""Hub-loading contract for the DPLM backbone wrapper."""

import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import models.backbone as backbone  # noqa: E402


class FakeNet:
    """Stand-in network carrying the attributes the wrapper copies."""

    def __init__(self):
        self.config = SimpleNamespace(hidden_size=8)
        self.tokenizer = "TOKENIZER"
        self.mask_id = 1
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
        self.x_id = 4


class FakeNetClass:
    """Records the pretrained-loader arguments."""

    calls = []

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        cls.calls.append((name, kwargs))
        return FakeNet()


class FakeAutoConfig:
    """Reports the only supported DPLM type."""

    @classmethod
    def from_pretrained(cls, name):
        return SimpleNamespace(dplm_type="dplm_esm")


def test_hub_branch_forwards_net_overrides_and_cfg_overrides(monkeypatch):
    """The hub branch resolves the type, loads the net and merges overrides."""
    FakeNetClass.calls = []
    monkeypatch.setattr(backbone, "AutoConfig", FakeAutoConfig)
    monkeypatch.setattr(backbone, "get_net_class", lambda _type: FakeNetClass)
    wrapper = backbone.DPLMBackbone.from_pretrained(
        "org/model",
        cfg_override={"net": {"dropout": 0.4}},
        net_override={"attn_implementation": "eager"},
    )
    assert FakeNetClass.calls == [("org/model", {"attn_implementation": "eager"})]
    assert isinstance(wrapper.net, FakeNet)
    assert isinstance(wrapper, backbone.DPLMBackbone)
    assert wrapper.cfg.net.dropout == 0.4
    assert wrapper.cfg.net.name == "esm2_t33_650M_UR50D"
    assert wrapper.tokenizer == "TOKENIZER"
