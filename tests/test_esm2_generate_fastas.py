"""Baseline/guided generation contract for the ESM2 evaluation workflow."""

import json
import os
import sys
import types
from pathlib import Path

import pytest
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


def build_config(tmp_path, **overrides):
    """Builds a real SA evaluation config rooted at ``tmp_path``."""
    arguments = {
        "reference_fasta": tmp_path / "ref.fasta",
        "proposal_ckpt": "proposal",
        "energy_ckpt": "energy",
        "guided_checkpoint": "guided.pt",
        "output_dir": tmp_path,
        "device": "cpu",
    }
    arguments.update(overrides)
    return esm2_eval.build_sa_eval_config(**arguments)


@pytest.fixture()
def recorded(monkeypatch):
    """Stubs the three collaborators and records their calls."""
    calls = {"build": [], "generate": [], "load": []}

    def fake_build(config, **kwargs):
        calls["build"].append(kwargs)
        return f"generator-{len(calls['build'])}"

    def fake_generate(generator, records, **kwargs):
        calls["generate"].append((generator, records, kwargs))
        return records

    def fake_load(generator, checkpoint, device):
        calls["load"].append((generator, checkpoint, device))

    monkeypatch.setattr(esm2_eval, "build_generator_for_eval", fake_build)
    monkeypatch.setattr(esm2_eval, "run_generation_over_records", fake_generate)
    monkeypatch.setattr(esm2_eval, "load_trained_energy_weights", fake_load)
    return calls


def test_missing_checkpoints_abort_before_any_build(tmp_path, recorded):
    """An unset guided checkpoint stops the workflow with the documented error."""
    config = build_config(tmp_path, guided_checkpoint="")
    with pytest.raises(SystemExit, match="must be set"):
        esm2_eval.generate_candidate_fastas(
            config=config,
            reference_records=[("h", "AC")],
            device=torch.device("cpu"),
            output_dir=tmp_path,
        )
    assert recorded["build"] == []


def test_baseline_and_guided_settings_are_kept_separate(tmp_path, recorded):
    """The baseline uses one candidate while the guided branch uses the config."""
    config = build_config(tmp_path)
    records = [("h", "AC")]
    device = torch.device("cpu")
    baseline_path, guided_path = esm2_eval.generate_candidate_fastas(
        config=config,
        reference_records=records,
        device=device,
        output_dir=tmp_path,
    )
    generation_dir = tmp_path / "generated"
    assert baseline_path == generation_dir / "baseline_generated_sequences.fasta"
    assert guided_path == generation_dir / "guided_generated_sequences.fasta"

    baseline_call, guided_call = recorded["build"]
    assert baseline_call == {"device": device, "num_candidates": 1}
    assert guided_call == {
        "device": device,
        "num_candidates": config.guided_num_candidates,
        "proposal_temperature": config.guided_proposal_temperature,
        "proposal_noise_scale": config.guided_proposal_noise_scale,
        "energy_temperature": config.guided_energy_temperature,
        "disable_resample": config.guided_disable_resample,
        "resample_ratio": config.guided_resample_ratio,
        "resample_top_p": config.guided_resample_top_p,
    }

    assert recorded["load"] == [("generator-2", "guided.pt", device)]
    baseline_run, guided_run = recorded["generate"]
    assert baseline_run[0] == "generator-1"
    assert baseline_run[1:] == (
        records,
        {
            "max_steps": config.generation_steps,
            "seed_base": config.seed,
            "output_fasta_path": baseline_path,
            "label": "baseline",
        },
    )
    assert guided_run[0] == "generator-2"
    assert guided_run[2]["output_fasta_path"] == guided_path
    assert guided_run[2]["label"] == "guided"


def test_generation_config_is_written_with_the_eval_settings(tmp_path, recorded):
    """The dumped JSON mirrors the generation-half of the eval config."""
    config = build_config(tmp_path)
    esm2_eval.generate_candidate_fastas(
        config=config,
        reference_records=[("h", "AC")],
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    payload = json.loads(
        (tmp_path / "generated" / "generation_config.json").read_text(encoding="utf-8")
    )
    assert payload["proposal_ckpt"] == "proposal"
    assert payload["energy_ckpt"] == "energy"
    assert payload["guided_checkpoint"] == "guided.pt"
    assert payload["generation_steps"] == config.generation_steps
    assert payload["seed"] == config.seed
    assert payload["guided_num_candidates"] == config.guided_num_candidates
    assert payload["bm_sampler_type"] == "sa"
    assert payload["bm_sampler_kwargs"] is None
