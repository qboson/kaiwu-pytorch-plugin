"""Checkpoint-payload helper contract for the DPLM runtime."""

import importlib
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion"))
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]

runtime = importlib.import_module("dplm.utils.runtime")


class DummyBM:
    """Returns a marker state dict."""

    def state_dict(self):
        return {"weight": 1.0}


def build_generator(**overrides):
    """Builds a generator whose energy model exposes the BM payload."""
    energy_model = SimpleNamespace(
        energy_bm=DummyBM(),
        sampler_type="cim",
        sampler_kwargs={"tmp_dir": "checkpoints"},
    )
    for key, value in overrides.items():
        setattr(energy_model, key, value)
    return SimpleNamespace(energy_model=energy_model)


def test_state_dict_is_wrapped_under_the_energy_bm_key():
    """The BM state dict is stored under ``energy_bm``."""
    payload = runtime._energy_backend_state(build_generator())
    assert payload == {"energy_bm": {"weight": 1.0}}


def test_metadata_reports_the_sampler_configuration():
    """Sampler type and kwargs are copied into the metadata payload."""
    metadata = runtime._energy_backend_metadata(build_generator())
    assert metadata == {
        "bm_sampler_type": "cim",
        "bm_sampler_kwargs": {"tmp_dir": "checkpoints"},
    }


def test_missing_sampler_attributes_fall_back_to_none_and_empty_dict():
    """A generator without sampler metadata still produces a usable payload."""
    generator = SimpleNamespace(
        energy_model=SimpleNamespace(energy_bm=DummyBM())
    )
    assert runtime._energy_backend_metadata(generator) == {
        "bm_sampler_type": None,
        "bm_sampler_kwargs": {},
    }
