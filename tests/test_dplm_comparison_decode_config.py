"""Regression test: baseline and guided generation must share decode settings.

The DPLM workflow's final comparison built the proposal-only baseline with the
builder defaults (proposal temperature 0.0, resample ratio 0.25, top-p 0.95)
while the guided branch passed the configured generation settings, so the
reported baseline-versus-guided differences mixed decoding changes in with the
effect of the learned reranker. Both comparison branches must receive the same
generation configuration; only the candidate count / reranking differs.
"""

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DPLM_DIR = REPO_ROOT / "example" / "qdiffusion" / "dplm"


def _load_workflow_train(monkeypatch):
    """Loads workflows/train.py with sibling imports stubbed."""
    stub_attrs = {
        "utils.dplm_builder": ("build_qdiffusion",),
        "utils.io": (
            "default_fasta_path",
            "default_outputs_root",
            "read_fasta_records",
            "save_json",
            "write_fasta_records",
        ),
        "utils.metrics": (
            "compare_generation_sets",
            "evaluate_generation_quality",
            "save_quality_summary",
        ),
        "utils.runtime": (
            "load_trained_energy_weights",
            "save_checkpoint",
            "seed_torch",
            "summarize_trainable_parameters",
        ),
        "workflow_helpers": (
            "build_data_loader_from_records",
            "run_epoch",
            "run_generation_over_records",
            "run_structural_validation",
            "select_records",
            "split_train_val_test",
            "write_markdown_report",
        ),
    }
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(DPLM_DIR / "utils")]
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    for name, attrs in stub_attrs.items():
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, lambda *a, **k: None)
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "workflow_train_under_test", DPLM_DIR / "workflows" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workflow_config(module, **generate_overrides):
    return module.WorkflowConfig(
        data=module.DataConfig(fasta_path="/tmp/unittest.fasta"),
        model=module.ModelConfig(proposal_ckpt="p", energy_ckpt="e"),
        generate=module.GenerateConfig(**generate_overrides),
    )


def test_baseline_and_guided_share_generation_settings(monkeypatch):
    module = _load_workflow_train(monkeypatch)
    recorded = []

    def _recording_builder(config, *, device, num_candidates, **kwargs):
        recorded.append({"num_candidates": num_candidates, **kwargs})
        return object()

    monkeypatch.setattr(module, "build_generator_from_config", _recording_builder)

    config = _workflow_config(
        module,
        proposal_temperature=0.4,
        proposal_noise_scale=0.7,
        energy_temperature=1.5,
        disable_resample=True,
        resample_ratio=0.11,
        resample_top_p=0.77,
        num_candidates=6,
    )
    baseline, guided = module.build_comparison_generators(config, device="cpu")

    assert len(recorded) == 2
    baseline_kwargs, guided_kwargs = recorded

    shared = (
        "proposal_temperature",
        "proposal_noise_scale",
        "energy_temperature",
        "disable_resample",
        "resample_ratio",
        "resample_top_p",
    )
    for key in shared:
        assert baseline_kwargs[key] == guided_kwargs[key] == getattr(
            config.generate, key
        ), f"comparison branches must share {key}"

    assert baseline_kwargs["num_candidates"] == 1
    assert guided_kwargs["num_candidates"] == config.generate.num_candidates
    assert baseline is not guided


def test_default_config_values_differ_from_the_builder_defaults(monkeypatch):
    """The bug matters: configured defaults are not the builder's fallbacks."""
    module = _load_workflow_train(monkeypatch)

    generate = module.GenerateConfig()
    signature_defaults = {
        "proposal_temperature": 0.0,
        "resample_ratio": 0.25,
        "resample_top_p": 0.95,
    }
    for key, builder_default in signature_defaults.items():
        assert getattr(generate, key) != builder_default
