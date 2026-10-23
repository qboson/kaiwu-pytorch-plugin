"""SA evaluation-config contract for the ESM2 workflow."""

import os
import sys
import types
from pathlib import Path

import pytest

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


def build(**overrides):
    """Builds one SA config with the required paths."""
    arguments = {
        "reference_fasta": Path("ref.fasta"),
        "proposal_ckpt": "proposal",
        "energy_ckpt": "energy",
        "guided_checkpoint": "guided.pt",
        "output_dir": Path("out"),
        "device": "cpu",
    }
    arguments.update(overrides)
    return esm2_eval.build_sa_eval_config(**arguments)


def test_required_paths_and_defaults_are_stored():
    """The config keeps the paths and the documented evaluation defaults."""
    config = build()
    assert isinstance(config, esm2_eval.EvalConfig)
    assert config.reference_fasta == Path("ref.fasta")
    assert config.proposal_ckpt == "proposal"
    assert config.energy_ckpt == "energy"
    assert config.guided_checkpoint == "guided.pt"
    assert config.output_dir == Path("out")
    assert config.device == "cpu"
    assert config.esm2_model == "esm2_t33_650M_UR50D"
    assert (config.pair_mode, config.pooling) == ("order", "mean")
    assert (config.batch_size, config.max_records) == (1, 20)
    assert (config.generation_steps, config.seed) == (500, 42)
    assert config.freeze_proposal is True
    assert config.guided_num_candidates == 4
    assert config.guided_proposal_temperature == pytest.approx(0.3)
    assert config.guided_proposal_noise_scale == pytest.approx(1.0)
    assert config.guided_energy_temperature == pytest.approx(1.25)
    assert config.guided_disable_resample is False
    assert config.guided_resample_ratio == pytest.approx(0.20)
    assert config.guided_resample_top_p == pytest.approx(0.90)


def test_sa_sampler_is_recorded_without_kwargs():
    """The SA config routes the BM through simulated annealing with no kwargs."""
    config = build()
    assert config.bm_sampler_type == "sa"
    assert config.bm_sampler_kwargs is None


def test_explicit_overrides_win_over_defaults():
    """Every keyword override replaces the corresponding default."""
    config = build(
        esm2_model="esm2_t12_35M_UR50D",
        pair_mode="header",
        pooling="max",
        batch_size=8,
        max_records=None,
        generation_steps=7,
        seed=3,
        freeze_proposal=False,
        guided_num_candidates=9,
        guided_proposal_temperature=0.5,
        guided_proposal_noise_scale=2.0,
        guided_energy_temperature=0.75,
        guided_disable_resample=True,
        guided_resample_ratio=0.4,
        guided_resample_top_p=0.7,
    )
    assert config.esm2_model == "esm2_t12_35M_UR50D"
    assert (config.pair_mode, config.pooling) == ("header", "max")
    assert (config.batch_size, config.max_records) == (8, None)
    assert (config.generation_steps, config.seed) == (7, 3)
    assert config.freeze_proposal is False
    assert config.guided_num_candidates == 9
    assert config.guided_proposal_temperature == pytest.approx(0.5)
    assert config.guided_proposal_noise_scale == pytest.approx(2.0)
    assert config.guided_energy_temperature == pytest.approx(0.75)
    assert config.guided_disable_resample is True
    assert config.guided_resample_ratio == pytest.approx(0.4)
    assert config.guided_resample_top_p == pytest.approx(0.7)
