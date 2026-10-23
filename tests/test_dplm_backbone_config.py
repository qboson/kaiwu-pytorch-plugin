"""Config-merge contract for the DPLM backbone wrapper."""

import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import models.backbone as backbone  # noqa: E402


class FakeNet:
    """Minimal wrapped network exposing the ids the wrapper copies."""

    def __init__(self):
        self.config = SimpleNamespace(hidden_size=16)
        self.tokenizer = "TOKENIZER"
        self.mask_id = 1
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
        self.x_id = 4
        self.supports_gradient_checkpointing = False
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True


def test_default_config_is_used_when_no_override_is_given():
    """The wrapper starts from the default DPLM config and copies the ids."""
    net = FakeNet()
    wrapper = backbone.DPLMBackbone(net=net)
    assert wrapper.cfg.net.name == "esm2_t33_650M_UR50D"
    assert wrapper.cfg.net.dropout == 0.1
    assert wrapper.cfg.num_diffusion_timesteps == 500
    assert wrapper.tokenizer == "TOKENIZER"
    assert (wrapper.mask_id, wrapper.pad_id, wrapper.bos_id) == (1, 0, 2)
    assert (wrapper.eos_id, wrapper.x_id) == (3, 4)
    assert net.checkpointing_enabled is False


def test_nested_override_merges_onto_the_defaults():
    """Overriding one nested field keeps the sibling defaults."""
    wrapper = backbone.DPLMBackbone(cfg={"net": {"dropout": 0.5}}, net=FakeNet())
    assert wrapper.cfg.net.dropout == 0.5
    assert wrapper.cfg.net.name == "esm2_t33_650M_UR50D"
    assert wrapper.cfg.net.arch_type == "esm"


def test_gradient_checkpointing_flag_enables_the_net_hook():
    """``gradient_ckpt`` flips the net flag and calls the enable hook."""
    net = FakeNet()
    wrapper = backbone.DPLMBackbone(cfg={"gradient_ckpt": True}, net=net)
    assert wrapper.cfg.gradient_ckpt is True
    assert net.supports_gradient_checkpointing is True
    assert net.checkpointing_enabled is True


def test_overrides_do_not_leak_between_instances():
    """The class-level default config is never mutated by one instance."""
    overridden = backbone.DPLMBackbone(cfg={"net": {"dropout": 0.7}}, net=FakeNet())
    fresh = backbone.DPLMBackbone(net=FakeNet())
    assert overridden.cfg.net.dropout == 0.7
    assert fresh.cfg.net.dropout == 0.1
