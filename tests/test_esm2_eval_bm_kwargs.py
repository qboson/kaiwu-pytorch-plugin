"""BM payload contract for the ESM2 evaluation workflow."""

import os
import sys
import types
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPLM = os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm")
sys.path[:0] = [DPLM, os.path.join(DPLM, "workflows")]

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
import esm2_eval  # noqa: E402
if _added:
    sys.modules.pop("esm", None)


def test_payload_forwards_the_configured_sampler():
    """The eval config's sampler type and kwargs are passed through unchanged."""
    kwargs = {"task_name": "demo", "wait": True}
    config = SimpleNamespace(bm_sampler_type="cim", bm_sampler_kwargs=kwargs)
    payload = esm2_eval.build_bm_kwargs(config)
    assert payload == {
        "bm_sampler_type": "cim",
        "bm_sampler_kwargs": {"task_name": "demo", "wait": True},
    }


def test_payload_keeps_a_missing_sampler_payload_as_none():
    """The SA config carries ``None`` rather than an empty dictionary."""
    config = SimpleNamespace(bm_sampler_type="sa", bm_sampler_kwargs=None)
    assert esm2_eval.build_bm_kwargs(config) == {
        "bm_sampler_type": "sa",
        "bm_sampler_kwargs": None,
    }
