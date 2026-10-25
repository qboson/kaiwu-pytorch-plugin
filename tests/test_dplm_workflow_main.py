"""Contract test for the ``workflows/train.py`` full-workflow entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
QDIFFUSION = REPO / "example" / "qdiffusion"


class _Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(2))
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)
        self.proposal_model = None
        self.train_calls = 0

    def train(self):
        self.train_calls += 1
        return self


class _Optimizer:
    instances: list["_Optimizer"] = []

    def __init__(self, parameters, **kwargs):
        self.parameters = list(parameters)
        self.kwargs = kwargs
        self.param_groups = [{"lr": kwargs["lr"]}]
        _Optimizer.instances.append(self)


class _Scheduler:
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.kwargs = kwargs
        self.steps: list[float] = []

    def step(self, value):
        self.steps.append(value)


def _config():
    return SimpleNamespace(
        train=SimpleNamespace(
            require_cuda=False,
            num_candidates=4,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.01,
            scheduler_factor=0.5,
            scheduler_patience=2,
            epochs=2,
            grad_clip_norm=1.0,
            min_epochs=1,
            early_stop_patience=2,
            validation_steps=3,
        ),
        data=SimpleNamespace(
            seed=7,
            fasta_path="/data/ref.fa",
            min_length=10,
            max_length=100,
            max_records=20,
            val_ratio=0.2,
            test_ratio=0.2,
        ),
        generate=SimpleNamespace(
            steps=5,
            num_candidates=2,
            proposal_temperature=0.3,
            proposal_noise_scale=1.0,
            energy_temperature=1.25,
            disable_resample=False,
            resample_ratio=0.2,
            resample_top_p=0.9,
        ),
    )


def _load_module():
    sys.path.insert(0, str(QDIFFUSION))
    try:
        from dplm.workflows import train as train_module
    finally:
        sys.path.remove(str(QDIFFUSION))
    return train_module


def test_full_workflow_wires_data_training_and_reports(tmp_path, monkeypatch, capsys):
    module = _load_module()
    config = _config()
    state: dict[str, object] = {
        "builds": [],
        "epochs": [],
        "checkpoints": [],
        "json": [],
        "fasta": [],
        "generations": [],
        "quality": [],
        "quality_files": [],
        "reports": [],
    }
    all_records = [(f"h{index}", "A" * 20) for index in range(5)]
    train_records = all_records[:3]
    val_records = all_records[3:4]
    test_records = all_records[4:]
    generator = _Generator()
    _Optimizer.instances.clear()

    module.build_default_workflow_config = lambda: config
    module.default_outputs_root = lambda: tmp_path
    module.seed_torch = lambda seed: state.setdefault("seed_torch", seed)
    monkeypatch.setattr(module.random, "seed", lambda seed: state.setdefault("seed_py", seed))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    module.asdict = lambda cfg: {"cfg": "stub"}
    module.read_fasta_records = lambda path: all_records
    def select_records(records, **kwargs):
        state["select_kwargs"] = kwargs
        return list(records)

    def split_records(records, **kwargs):
        state["split_kwargs"] = kwargs
        return train_records, val_records, test_records

    module.select_records = select_records
    module.split_train_val_test = split_records
    module.save_json = lambda path, payload: state["json"].append((path, payload))
    module.write_fasta_records = lambda path, records: state["fasta"].append(
        (path, list(records))
    )

    def build_generator(config_arg, **kwargs):
        state["builds"].append((config_arg, kwargs))
        return _Generator()

    module.build_generator_from_config = build_generator
    module.run_structural_validation = lambda gen, record, **kwargs: (
        state.setdefault("validation", (gen, record, kwargs)) or {"ok": 1}
    )
    module.summarize_trainable_parameters = lambda gen: {"total_parameters": 2}
    module.build_data_loader_from_records = lambda gen, records, **kwargs: (
        state.setdefault("loaders", []).append((gen, records, kwargs))
        or f"loader-{kwargs['shuffle']}"
    )
    module.AdamW = _Optimizer
    module.ReduceLROnPlateau = _Scheduler

    metric_table = iter(
        [
            {"energy_objective_mean": 1.0},
            {"energy_objective_mean": 0.5},
            {"energy_objective_mean": 0.9},
            {"energy_objective_mean": 0.4},
        ]
    )
    module.run_epoch = lambda gen, loader, **kwargs: (
        state["epochs"].append((loader, kwargs)) or next(metric_table)
    )

    def save_checkpoint(directory, name, **kwargs):
        state["checkpoints"].append((directory, name, kwargs))
        return directory / name

    module.save_checkpoint = save_checkpoint

    def generate_records(gen, records, **kwargs):
        state["generations"].append((gen, records, kwargs))
        return [("g", "SEQ")], {"records": [("g", "SEQ")]}

    module.run_generation_over_records = generate_records
    module.load_trained_energy_weights = lambda gen, path, device: state.setdefault(
        "loaded", (gen, path, device)
    )
    module.evaluate_generation_quality = lambda reference, candidate, label: (
        state["quality"].append((reference, candidate, label)) or {"label": label}
    )
    module.save_quality_summary = lambda path, quality: state["quality_files"].append(
        (path, quality)
    )
    module.compare_generation_sets = lambda reference, baseline, guided: {
        "compared": (len(reference), len(baseline), len(guided))
    }
    module.write_markdown_report = lambda path, **kwargs: state["reports"].append(
        (path, kwargs)
    )

    module.main()

    assert state["seed_py"] == 7
    assert state["seed_torch"] == 7
    assert state["select_kwargs"] == {
        "min_length": 10,
        "max_length": 100,
        "max_records": 20,
    }
    assert state["split_kwargs"] == {
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "seed": 7,
    }
    run_config = state["json"][0]
    assert run_config[0].name == "run_config.json"
    assert run_config[1]["device"] == "cpu"
    assert run_config[1]["all_records"] == 5
    assert run_config[1]["selected_records"] == 5
    assert run_config[1]["train_records"] == 3
    assert run_config[1]["val_records"] == 1
    assert run_config[1]["test_records"] == 1
    assert [path.name for path, _ in state["fasta"]] == [
        "train.fasta",
        "val.fasta",
        "test.fasta",
    ]
    assert [records for _, records in state["fasta"]] == [
        train_records,
        val_records,
        test_records,
    ]
    assert [kwargs["num_candidates"] for _, kwargs in state["builds"]] == [1, 4, 1, 2]
    validation_generator, validation_record, validation_kwargs = state["validation"]
    assert validation_record == test_records[0]
    assert validation_kwargs == {"max_length": 100, "steps": 3}
    assert state["loaders"][0][2] == {"batch_size": 2, "shuffle": True}
    assert state["loaders"][1][2] == {"batch_size": 2, "shuffle": False}
    optimizer = _Optimizer.instances[0]
    assert len(optimizer.parameters) == 1
    assert optimizer.kwargs == {"lr": 1e-3, "weight_decay": 0.01}
    assert [kwargs for _, kwargs in state["epochs"]] == [
        {"optimizer": optimizer, "grad_clip_norm": 1.0, "description": "train-1"},
        {"optimizer": None, "grad_clip_norm": 1.0, "description": "val-1"},
        {"optimizer": optimizer, "grad_clip_norm": 1.0, "description": "train-2"},
        {"optimizer": None, "grad_clip_norm": 1.0, "description": "val-2"},
    ]
    assert [name for _, name, _ in state["checkpoints"]] == [
        "best_epoch_1.pt",
        "best_epoch_2.pt",
        "final_epoch_2.pt",
    ]
    history = [
        payload for path, payload in state["json"] if path.name == "history.json"
    ][-1]
    assert isinstance(history, list) and history[-1]["epoch"] == 2
    assert [call["label"] for _, _, call in state["generations"]] == [
        "proposal_only",
        "energy_guided",
    ]
    assert state["generations"][1][2]["output_dir"].name == "guided"
    loaded_generator, loaded_path, loaded_device = state["loaded"]
    assert loaded_device == "cpu"
    assert Path(loaded_path).name == "best_epoch_2.pt"
    assert [label for _, _, label in state["quality"]] == [
        "proposal_only",
        "energy_guided",
    ]
    assert [path.name for path, _ in state["quality_files"]] == [
        "baseline_eval",
        "guided_eval",
    ]
    report_path, report_kwargs = state["reports"][0]
    assert report_path.name == "REPORT.md"
    assert report_kwargs["selected_count"] == 5
    assert report_kwargs["train_count"] == 3
    assert report_kwargs["test_count"] == 1
    assert "Real full workflow completed." in capsys.readouterr().out
