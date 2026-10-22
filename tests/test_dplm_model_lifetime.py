"""Run the real workflow with tiny models and assert phase-local ownership."""

import ast
import json
import random
import weakref
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def test_previous_phase_models_are_released_before_building_next(tmp_path):
    path = Path(__file__).resolve().parents[1] / "example/qdiffusion/dplm/workflows/train.py"
    parsed = ast.parse(path.read_text())
    main = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    module = ast.Module(body=[main], type_ignores=[])
    ast.fix_missing_locations(module)
    references = []
    optimizer_references = []
    generations = []

    class TinyGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.proposal_model = nn.Linear(1, 1)

    class Loader(list):
        def __init__(self, generator):
            super().__init__()
            self.generator = generator

    def build(*args, **kwargs):
        assert all(ref() is None for ref in references), "previous phase model remains alive"
        if len(references) >= 2:
            assert all(ref() is None for ref in optimizer_references), "training optimizer remains alive"
        model = TinyGenerator()
        references.append(weakref.ref(model))
        return model

    def optimizer(*args, **kwargs):
        result = AdamW(*args, **kwargs)
        optimizer_references.append(weakref.ref(result))
        return result

    def generation(model, records, **kwargs):
        generations.append(kwargs["label"])
        return list(records), []

    config = SimpleNamespace(
        data=SimpleNamespace(seed=42, fasta_path="unused", min_length=1, max_length=4,
                             max_records=None, val_ratio=.2, test_ratio=.2),
        train=SimpleNamespace(require_cuda=False, num_candidates=2, validation_steps=1,
                              batch_size=1, learning_rate=.01, weight_decay=0.,
                              scheduler_factor=.5, scheduler_patience=1, epochs=1,
                              grad_clip_norm=1., min_epochs=1, early_stop_patience=4),
        generate=SimpleNamespace(steps=1, num_candidates=2, proposal_temperature=.3,
                                 proposal_noise_scale=1., energy_temperature=1.,
                                 disable_resample=True, resample_ratio=.2, resample_top_p=.9),
    )
    records = [("a", "AC"), ("b", "DE"), ("c", "FG")]
    checkpoints = []

    def save_checkpoint(directory, name, **kwargs):
        checkpoints.append(name)
        return directory / name

    namespace = dict(
        torch=torch, Path=Path, datetime=datetime, random=random, json=json,
        asdict=lambda config: {},
        build_default_workflow_config=lambda: config, default_outputs_root=lambda: tmp_path,
        seed_torch=torch.manual_seed, read_fasta_records=lambda path: records,
        select_records=lambda records, **kw: records,
        split_train_val_test=lambda records, **kw: (records[:1], records[1:2], records[2:]),
        save_json=lambda *a: None, write_fasta_records=lambda *a: None,
        build_generator_from_config=build, run_structural_validation=lambda *a, **kw: {},
        summarize_trainable_parameters=lambda model: {},
        build_data_loader_from_records=lambda model, *a, **kw: Loader(model),
        AdamW=optimizer, ReduceLROnPlateau=ReduceLROnPlateau,
        run_epoch=lambda *a, **kw: {"energy_objective_mean": 1.},
        save_checkpoint=save_checkpoint, run_generation_over_records=generation,
        load_trained_energy_weights=lambda *a: {},
        evaluate_generation_quality=lambda *a, **kw: {}, save_quality_summary=lambda *a: None,
        compare_generation_sets=lambda *a: {}, write_markdown_report=lambda *a, **kw: None,
    )
    exec(compile(module, str(path), "exec"), namespace)
    namespace["main"]()
    assert len(references) == 4
    assert all(ref() is None for ref in references)
    assert generations == ["proposal_only", "energy_guided"]
    assert checkpoints == ["best_epoch_1.pt", "final_epoch_1.pt"]
