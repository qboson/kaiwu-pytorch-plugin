# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Training workflow for running full-corpus ``QDiffusion`` experiments."""

from __future__ import annotations

import os
import random
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

try:
    from .._example_bootstrap import ensure_repo_src_on_path
except ImportError:  # pragma: no cover - direct script-path compatibility
    _WORKFLOW_DIR = Path(__file__).resolve().parent
    _CASE_DIR = _WORKFLOW_DIR.parent
    if str(_CASE_DIR) not in sys.path:
        sys.path.insert(0, str(_CASE_DIR))
    from _example_bootstrap import ensure_repo_src_on_path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

ensure_repo_src_on_path()

try:
    from ..utils.qdiffusion_protein_builder import build_qdiffusion
    from ..utils.io import (
        default_fasta_path,
        default_outputs_root,
        read_fasta_records,
        save_json,
        write_fasta_records,
    )
    from ..utils.metrics import (
        compare_generation_sets,
        evaluate_generation_quality,
        save_quality_summary,
    )
    from ..utils.runtime import (
        load_trained_energy_weights,
        save_checkpoint,
        seed_torch,
        summarize_trainable_parameters,
    )
    from .workflow_helpers import (
        build_data_loader_from_records,
        run_epoch,
        run_generation_over_records,
        run_structural_validation,
        select_records,
        split_train_val_test,
        write_markdown_report,
    )
except ImportError:  # pragma: no cover - direct script-path compatibility
    from utils.qdiffusion_protein_builder import build_qdiffusion
    from utils.io import (
        default_fasta_path,
        default_outputs_root,
        read_fasta_records,
        save_json,
        write_fasta_records,
    )
    from utils.metrics import (
        compare_generation_sets,
        evaluate_generation_quality,
        save_quality_summary,
    )
    from utils.runtime import (
        load_trained_energy_weights,
        save_checkpoint,
        seed_torch,
        summarize_trainable_parameters,
    )
    from workflow_helpers import (
        build_data_loader_from_records,
        run_epoch,
        run_generation_over_records,
        run_structural_validation,
        select_records,
        split_train_val_test,
        write_markdown_report,
    )

os.environ.setdefault("BYPROT_EAGER_IMPORTS", "0")


def build_generator_from_config(
    config: "WorkflowConfig",
    *,
    device: str,
    num_candidates: int,
    proposal_temperature: float = 0.0,
    proposal_noise_scale: float = 1.0,
    energy_temperature: float = 1.0,
    disable_resample: bool = False,
    resample_ratio: float = 0.25,
    resample_top_p: float = 0.95,
) -> Any:
    """Builds one DPLM-backed generator from the shared workflow config.

    The train, validation, baseline, and guided branches all reuse the same
    builder with different decode-time overrides. Keeping that mapping in one
    helper makes it easier to see which knobs are shared versus branch-specific.
    """
    return (
        build_qdiffusion(
            proposal_ckpt=config.model.proposal_ckpt,
            energy_ckpt=config.model.energy_ckpt,
            num_candidates=num_candidates,
            proposal_temperature=proposal_temperature,
            proposal_noise_scale=proposal_noise_scale,
            energy_temperature=energy_temperature,
            disable_resample=disable_resample,
            resample_ratio=resample_ratio,
            resample_top_p=resample_top_p,
            freeze_proposal=config.model.freeze_proposal,
            **make_bm_build_kwargs(config),
        )
        .eval()
        .to(device)
    )


@dataclass
class DataConfig:
    """Dataset selection and split settings."""

    fasta_path: str
    min_length: int = 50
    max_length: int = 256
    max_records: int | None = None
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42


@dataclass
class ModelConfig:
    """Model checkpoints shared by training and generation."""

    proposal_ckpt: str
    energy_ckpt: str
    freeze_proposal: bool = True


@dataclass
class SamplerConfig:
    """Energy sampler settings.

    Most runs only need ``sampler_type="sa"``. ``sampler_kwargs`` stays empty
    unless the example is routed through a custom sampler such as CIM.
    """

    sampler_type: str = "sa"
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Training-only knobs."""

    epochs: int = 20
    min_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    num_candidates: int = 4
    validation_steps: int = 3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1
    early_stop_patience: int = 4
    require_cuda: bool = True


@dataclass
class GenerateConfig:
    """Generation/evaluation knobs used after training."""

    num_candidates: int = 8
    energy_temperature: float = 1.25
    proposal_temperature: float = 0.3
    proposal_noise_scale: float = 1.0
    disable_resample: bool = False
    resample_ratio: float = 0.20
    resample_top_p: float = 0.90
    steps: int = 5


@dataclass
class WorkflowConfig:
    """Top-level config grouped by concern instead of one flat field list."""

    data: DataConfig
    model: ModelConfig
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)


def make_bm_build_kwargs(config: WorkflowConfig) -> dict[str, Any]:
    """Builds the small BM config payload consumed by the generator builder."""
    return {
        "bm_sampler_type": config.sampler.sampler_type,
        "bm_sampler_kwargs": dict(config.sampler.sampler_kwargs),
    }


def build_default_workflow_config() -> WorkflowConfig:
    """Returns one short, example-friendly default config."""
    return WorkflowConfig(
        data=DataConfig(fasta_path=str(default_fasta_path())),
        model=ModelConfig(
            proposal_ckpt="airkingbd/dplm_150m",
            energy_ckpt="airkingbd/dplm_150m",
        ),
        sampler=SamplerConfig(
            sampler_type="sa",
        ),
    )


def main() -> None:
    """Runs the final self-contained full workflow."""
    config = build_default_workflow_config()
    # For custom samplers, keep the top-level structure and only swap this block.
    #
    # config.sampler = SamplerConfig(
    #     sampler_type="cim",
    #     sampler_kwargs={
    #         "task_name": "qdiffusion_bm",
    #         "project_no": "YOUR_PROJECT_ID",
    #         "task_mode": "OPTIMIZATION",
    #         "tmp_dir": "./tmp",
    #         "wait": False,
    #     },
    # )

    has_cuda = torch.cuda.is_available()
    if config.train.require_cuda and not has_cuda:
        raise RuntimeError("CUDA is required for this real full-corpus workflow.")
    device = "cuda" if has_cuda else "cpu"

    output_root = default_outputs_root()
    run_name = datetime.now().strftime("real_full_workflow_%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")

    random.seed(config.data.seed)
    seed_torch(config.data.seed)

    # Stage 1: read the reference FASTA, filter it, and create deterministic
    # train/validation/test splits for the rest of the run.
    all_records = read_fasta_records(Path(config.data.fasta_path))
    selected_records = select_records(
        all_records,
        min_length=config.data.min_length,
        max_length=config.data.max_length,
        max_records=config.data.max_records,
    )
    train_records, val_records, test_records = split_train_val_test(
        selected_records,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.seed,
    )

    save_json(
        output_dir / "run_config.json",
        {
            **asdict(config),
            "device": device,
            "all_records": len(all_records),
            "selected_records": len(selected_records),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records": len(test_records),
        },
    )

    splits_dir = output_dir / "data_splits"
    write_fasta_records(splits_dir / "train.fasta", train_records)
    write_fasta_records(splits_dir / "val.fasta", val_records)
    write_fasta_records(splits_dir / "test.fasta", test_records)

    # Stage 2: run one pre-training structural sanity check so we can catch
    # tokenization / generation wiring issues before the expensive training loop.
    validation_generator = build_generator_from_config(
        config,
        device=device,
        num_candidates=1,
    )
    # Before spending time on training, verify that one real sequence can pass
    # through both ``objective`` and ``generate`` without shape/wiring issues.
    validation_summary = run_structural_validation(
        validation_generator,
        test_records[0],
        max_length=config.data.max_length,
        steps=config.train.validation_steps,
    )
    save_json(output_dir / "validation_summary.json", validation_summary)

    # Stage 3: build the actual trainable generator and dataloaders. In the
    # default setup, proposal weights stay frozen and the energy side is tuned.
    train_generator = build_generator_from_config(
        config,
        device=device,
        num_candidates=config.train.num_candidates,
    )
    train_generator.train()  # torch train()
    if getattr(train_generator, "proposal_model", None) is not None:
        train_generator.proposal_model.eval()
    save_json(
        output_dir / "parameter_summary.json",
        summarize_trainable_parameters(train_generator),
    )

    train_loader = build_data_loader_from_records(
        train_generator,
        train_records,
        batch_size=config.train.batch_size,
        shuffle=True,
    )
    val_loader = build_data_loader_from_records(
        train_generator,
        val_records,
        batch_size=config.train.batch_size,
        shuffle=False,
    )

    optimizer = AdamW(
        [
            parameter
            for parameter in train_generator.parameters()
            if parameter.requires_grad
        ],
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.train.scheduler_factor,
        patience=config.train.scheduler_patience,
    )

    checkpoints_dir = output_dir / "checkpoints"
    train_history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_checkpoint_path: Path | None = None
    epochs_without_improvement = 0

    # Stage 4: optimize the energy-guided objective, track validation loss, and
    # keep the best energy-side checkpoint for later guided generation.
    for epoch in range(1, config.train.epochs + 1):
        print(f"Epoch {epoch}/{config.train.epochs}")
        train_metrics = run_epoch(
            train_generator,
            train_loader,
            optimizer=optimizer,
            grad_clip_norm=config.train.grad_clip_norm,
            description=f"train-{epoch}",
        )
        val_metrics = run_epoch(
            train_generator,
            val_loader,
            optimizer=None,
            grad_clip_norm=config.train.grad_clip_norm,
            description=f"val-{epoch}",
        )

        epoch_summary = {
            "epoch": epoch,
            "train_energy_objective_mean": train_metrics["energy_objective_mean"],
            "val_energy_objective_mean": val_metrics["energy_objective_mean"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_positive_energy_mean": train_metrics.get(
                "positive_energy_mean", 0.0
            ),
            "train_negative_energy_mean": train_metrics.get(
                "negative_energy_mean", 0.0
            ),
            "val_positive_energy_mean": val_metrics.get("positive_energy_mean", 0.0),
            "val_negative_energy_mean": val_metrics.get("negative_energy_mean", 0.0),
            "sampling_mode": train_metrics.get("positive_sampling_mode", 0.0),
            "train_visible_on_ratio": train_metrics.get(
                "positive_visible_on_ratio", 0.0
            ),
            "train_hidden_on_ratio": train_metrics.get("positive_hidden_on_ratio", 0.0),
            "val_visible_on_ratio": val_metrics.get("positive_visible_on_ratio", 0.0),
            "val_hidden_on_ratio": val_metrics.get("positive_hidden_on_ratio", 0.0),
        }
        train_history.append(epoch_summary)
        save_json(output_dir / "history.json", train_history)
        print(json.dumps(epoch_summary, ensure_ascii=False))

        current_val = val_metrics["energy_objective_mean"]
        scheduler.step(current_val)
        if current_val < best_val:
            best_val = current_val
            epochs_without_improvement = 0
            best_checkpoint_path = save_checkpoint(
                checkpoints_dir,
                f"best_epoch_{epoch}.pt",
                generator=train_generator,
                epoch=epoch,
                metric=best_val,
            )
        else:
            epochs_without_improvement += 1

        if (
            epoch >= config.train.min_epochs
            and epochs_without_improvement >= config.train.early_stop_patience
        ):
            print(
                "Early stopping triggered: "
                f"no validation improvement for {epochs_without_improvement} epochs."
            )
            break

    final_epoch = train_history[-1]["epoch"]
    final_checkpoint_path = save_checkpoint(
        checkpoints_dir,
        f"final_epoch_{final_epoch}.pt",
        generator=train_generator,
        epoch=final_epoch,
        metric=train_history[-1]["val_energy_objective_mean"],
    )
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")

    # Stage 5: compare a proposal-only baseline against a guided generator that
    # reloads the best learned energy weights from training.
    baseline_generator = build_generator_from_config(
        config,
        device=device,
        num_candidates=1,
    )
    # The baseline branch keeps the proposal path only; it gives us a direct
    # comparison point for how much the learned BM reranker helps.
    baseline_records, _ = run_generation_over_records(
        baseline_generator,
        test_records,
        max_steps=config.generate.steps,
        seed_base=config.data.seed,
        output_dir=output_dir / "baseline",
        label="proposal_only",
    )

    guided_generator = build_generator_from_config(
        config,
        device=device,
        num_candidates=config.generate.num_candidates,
        proposal_temperature=config.generate.proposal_temperature,
        proposal_noise_scale=config.generate.proposal_noise_scale,
        energy_temperature=config.generate.energy_temperature,
        disable_resample=config.generate.disable_resample,
        resample_ratio=config.generate.resample_ratio,
        resample_top_p=config.generate.resample_top_p,
    )
    if best_checkpoint_path is None:
        raise RuntimeError("Training did not produce a best checkpoint.")
    # Guided generation reloads only the trained energy-side weights onto a
    # fresh generator instance so inference mirrors rerun/eval behavior.
    load_trained_energy_weights(guided_generator, str(best_checkpoint_path), device)
    guided_records, _ = run_generation_over_records(
        guided_generator,
        test_records,
        max_steps=config.generate.steps,
        seed_base=config.data.seed,
        output_dir=output_dir / "guided",
        label="energy_guided",
    )

    baseline_quality = evaluate_generation_quality(
        test_records, baseline_records, label="proposal_only"
    )
    guided_quality = evaluate_generation_quality(
        test_records, guided_records, label="energy_guided"
    )
    save_quality_summary(output_dir / "baseline_eval", baseline_quality)
    save_quality_summary(output_dir / "guided_eval", guided_quality)

    # Stage 6: aggregate the two generation branches into machine-readable and
    # human-readable reports for later inspection.
    comparison_summary = compare_generation_sets(
        test_records, baseline_records, guided_records
    )
    save_json(output_dir / "baseline_vs_guided.json", comparison_summary)

    write_markdown_report(
        output_dir / "REPORT.md",
        config=config,
        validation_summary=validation_summary,
        train_history=train_history,
        baseline_quality=baseline_quality,
        guided_quality=guided_quality,
        comparison_summary=comparison_summary,
        selected_count=len(selected_records),
        train_count=len(train_records),
        val_count=len(val_records),
        test_count=len(test_records),
        best_checkpoint_path=best_checkpoint_path,
    )

    print("Real full workflow completed.")
    print(f"Report: {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
