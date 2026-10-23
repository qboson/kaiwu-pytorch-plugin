"""Generator-mapping contract for the ESM2 evaluation workflow."""

import os
import sys
import types
from types import SimpleNamespace

import torch

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


class ChainableGenerator:
    """Records the builder kwargs and the eval/to chaining."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chain = []
        ChainableGenerator.instances.append(self)

    def eval(self):
        self.chain.append("eval")
        return self

    def to(self, device):
        self.chain.append(("to", device))
        return self


def make_config(**overrides):
    """Builds the config attributes the eval builder reads."""
    values = {
        "proposal_ckpt": "proposal",
        "energy_ckpt": "energy",
        "freeze_proposal": True,
        "bm_sampler_type": "cim",
        "bm_sampler_kwargs": {"task_name": "eval"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_generator_is_built_with_the_shared_and_branch_settings(monkeypatch):
    """Branch overrides, freeze flag and BM payload all reach the builder."""
    ChainableGenerator.instances = []
    monkeypatch.setattr(esm2_eval, "build_qdiffusion", ChainableGenerator)
    device = torch.device("cpu")
    generator = esm2_eval.build_generator_for_eval(
        make_config(),
        device=device,
        num_candidates=3,
        proposal_temperature=0.2,
        proposal_noise_scale=1.5,
        energy_temperature=0.75,
        disable_resample=True,
        resample_ratio=0.4,
        resample_top_p=0.8,
    )
    assert isinstance(generator, ChainableGenerator)
    kwargs = generator.kwargs
    assert kwargs["proposal_ckpt"] == "proposal"
    assert kwargs["energy_ckpt"] == "energy"
    assert kwargs["num_candidates"] == 3
    assert kwargs["proposal_temperature"] == 0.2
    assert kwargs["proposal_noise_scale"] == 1.5
    assert kwargs["energy_temperature"] == 0.75
    assert kwargs["disable_resample"] is True
    assert kwargs["resample_ratio"] == 0.4
    assert kwargs["resample_top_p"] == 0.8
    assert kwargs["freeze_proposal"] is True
    assert kwargs["bm_sampler_type"] == "cim"
    assert kwargs["bm_sampler_kwargs"] == {"task_name": "eval"}
    assert generator.chain == ["eval", ("to", device)]


def test_defaults_keep_proposal_only_decoding_settings(monkeypatch):
    """The baseline defaults keep one greedy candidate and no branch overrides."""
    ChainableGenerator.instances = []
    monkeypatch.setattr(esm2_eval, "build_qdiffusion", ChainableGenerator)
    generator = esm2_eval.build_generator_for_eval(
        make_config(freeze_proposal=False),
        device=torch.device("cpu"),
        num_candidates=1,
    )
    kwargs = generator.kwargs
    assert kwargs["num_candidates"] == 1
    assert kwargs["proposal_temperature"] == 0.0
    assert kwargs["proposal_noise_scale"] == 1.0
    assert kwargs["energy_temperature"] == 1.0
    assert kwargs["disable_resample"] is False
    assert kwargs["resample_ratio"] == 0.25
    assert kwargs["resample_top_p"] == 0.95
    assert kwargs["freeze_proposal"] is False
