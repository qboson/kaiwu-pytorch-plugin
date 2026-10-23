"""Backbone-class selection contract for the DPLM example."""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import models.backbone as backbone  # noqa: E402


def test_supported_dplm_type_returns_the_patched_esm_class():
    """The only supported family maps to the private patched ESM class."""
    assert backbone.get_net_class("dplm_esm") is backbone._EsmForDPLM


def test_other_types_are_rejected_with_the_type_named():
    """Unsupported or misspelled types fail with a message naming the value."""
    with pytest.raises(ValueError, match="Unsupported dplm_type"):
        backbone.get_net_class("esm")
    with pytest.raises(ValueError, match="dplm_esm "):
        backbone.get_net_class("dplm_esm ")
