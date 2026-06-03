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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ensure_repo_src_on_path()

try:
    from ..qdiffusion_protein_builder import build_qdiffusion_protein
    from ..utils.io import (
        default_fasta_path,
        default_outputs_root,
        normalize_decoded_sequence,
        read_fasta_records,
        save_json,
        save_markdown,
        write_fasta_records,
        write_tsv_rows,
    )
    from ..utils.metrics import (
        QualitySummary,
        compare_generation_sets,
        evaluate_generation_quality,
        save_quality_summary,
    )
    from ..utils.runtime import (
        encode_sequence,
        load_trained_energy_weights,
        save_checkpoint,
        seed_torch,
        summarize_trainable_parameters,
    )
except ImportError:  # pragma: no cover - direct script-path compatibility
    from qdiffusion_protein_builder import build_qdiffusion_protein
    from utils.io import (
        default_fasta_path,
        default_outputs_root,
        normalize_decoded_sequence,
        read_fasta_records,
        save_json,
        save_markdown,
        write_fasta_records,
        write_tsv_rows,
    )
    from utils.metrics import (
        QualitySummary,
        compare_generation_sets,
        evaluate_generation_quality,
        save_quality_summary,
    )
    from utils.runtime import (
        encode_sequence,
        load_trained_energy_weights,
        save_checkpoint,
        seed_torch,
        summarize_trainable_parameters,
    )

os.environ.setdefault("BYPROT_EAGER_IMPORTS", "0")


def select_records(
    records: list[tuple[str, str]],
    *,
    min_length: int,
    max_length: int,
    max_records: int | None,
) -> list[tuple[str, str]]:
    """Filters records by length and optional count limit.

    Args:
        records: Input FASTA records.
        min_length: Minimum acceptable sequence length.
        max_length: Maximum acceptable sequence length.
        max_records: Optional maximum number of records to keep.

    Returns:
        list[tuple[str, str]]: Filtered FASTA records.
    """
    selected: list[tuple[str, str]] = []
    for header, sequence in records:
        if len(sequence) < min_length or len(sequence) > max_length:
            continue
        selected.append((header, sequence))
        if max_records is not None and len(selected) >= max_records:
            break
    return selected


def split_train_val_test(
    records: list[tuple[str, str]],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Creates deterministic train/validation/test splits.

    Args:
        records: Filtered FASTA records.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        seed: Random seed for deterministic shuffling.

    Returns:
        tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]: A tuple ``(train_records, val_records, test_records)``.

    Raises:
        ValueError: If the dataset is too small or split ratios consume all
            records.
    """
    if len(records) < 3:
        raise ValueError("Need at least 3 records to build train/val/test splits.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio))
    test_size = max(1, int(len(shuffled) * test_ratio))
    if val_size + test_size >= len(shuffled):
        raise ValueError("Validation and test splits consumed every record.")

    val_records = shuffled[:val_size]
    test_records = shuffled[val_size : val_size + test_size]
    train_records = shuffled[val_size + test_size :]
    return train_records, val_records, test_records


def summarize_objective(outputs: dict[str, torch.Tensor]) -> str:
    """Formats the main structural tensors returned by ``objective()``.

    Args:
        outputs: Output dictionary returned by ``generator.objective(...)``.

    Returns:
        str: Human-readable one-line summary for logging.
    """
    logits = outputs["logits"]
    targets = outputs["targets"]
    loss_mask = outputs["loss_mask"]
    weight = outputs["weight"]
    energy_objective = outputs["energy_objective"]
    return (
        f"logits={tuple(logits.shape)}, "
        f"targets={tuple(targets.shape)}, "
        f"masked_positions={int(loss_mask.sum().item())}, "
        f"weight_mean={float(weight.mean().item()):.4f}, "
        f"energy_objective_mean={float(energy_objective.mean().item()):.4f}, "
        f"positive_energy_mean={float(outputs.get('positive_energy_mean', torch.tensor(0.0)).item()):.4f}, "
        f"negative_energy_mean={float(outputs.get('negative_energy_mean', torch.tensor(0.0)).item()):.4f}"
    )


class FastaSequenceDataset(Dataset[dict[str, str]]):
    """Simple dataset wrapper around FASTA records."""

    def __init__(self, records: list[tuple[str, str]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        header, sequence = self.records[index]
        return {"header": header, "sequence": sequence}


def make_collate_fn(generator):
    """Builds a generator-bound tokenization collate function.

    Args:
        generator: Configured ``QDiffusion`` instance.

    Returns:
        A collate function that tokenizes FASTA batches on the generator device.
    """

    def collate(batch: list[dict[str, str]]) -> dict[str, Any]:
        sequences = [item["sequence"] for item in batch]
        headers = [item["header"] for item in batch]
        encoded = generator.tokenizer(
            sequences,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        )
        return {
            "headers": headers,
            "targets": encoded["input_ids"].to(generator.device),
        }

    return collate


def build_data_loader_from_records(
    generator, records, *, batch_size: int, shuffle: bool
) -> DataLoader:
    """Creates one dataloader from explicit FASTA records.

    Args:
        generator: Configured ``QDiffusion`` instance.
        records: FASTA records to load.
        batch_size: Batch size for the dataloader.
        shuffle: Whether to shuffle records each epoch.

    Returns:
        DataLoader: A dataloader producing tokenized batches.
    """
    return DataLoader(
        FastaSequenceDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(generator),
    )


def set_training_mode(generator, training: bool) -> None:
    """Toggles train/eval while keeping the proposal backbone frozen.

    Args:
        generator: Configured ``QDiffusion`` instance.
        training: Whether the surrounding loop is in training mode.
    """
    if training:
        generator.train()
        if getattr(generator, "proposal_model", None) is not None:
            generator.proposal_model.eval()
    else:
        generator.eval()


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
        build_qdiffusion_protein(
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


def run_epoch(
    generator,
    data_loader: DataLoader,
    *,
    optimizer: AdamW | None,
    grad_clip_norm: float,
    description: str,
) -> dict[str, float]:
    """Runs one training or validation epoch.

    Args:
        generator: Configured ``QDiffusion`` instance.
        data_loader: Dataloader of tokenized FASTA batches.
        optimizer: Optimizer used for training. When ``None``, run evaluation.
        grad_clip_norm: Gradient clipping threshold.
        description: Progress-bar label.

    Returns:
        dict[str, float]: Aggregated epoch metrics.
    """
    training = optimizer is not None
    set_training_mode(generator, training=training)

    total_loss = 0.0
    total_examples = 0
    metric_totals: dict[str, float] = {}
    tracked_keys = (
        "positive_energy_mean",
        "negative_energy_mean",
        "positive_sampling_mode",
        "negative_sampling_mode",
        "positive_visible_on_ratio",
        "negative_visible_on_ratio",
        "positive_hidden_on_ratio",
        "negative_hidden_on_ratio",
    )

    # Shared epoch loop: with an optimizer we train, without one we only run
    # the same objective in evaluation mode for validation statistics.
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(data_loader, desc=description, unit="batch"):
            # ``objective(...)`` packages the whole one-step diffusion training
            # computation and returns the energy-guided objective we optimize.
            outputs = generator.objective({"targets": batch["targets"]})
            loss = outputs["energy_objective"].mean()

            if training:
                optimizer.zero_grad(set_to_none=True)
                # Gradients only update the energy side in the default example
                # setup because the proposal backbone is kept frozen.
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in generator.parameters() if p.requires_grad],
                        grad_clip_norm,
                    )
                optimizer.step()

            batch_size = int(batch["targets"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            for key in tracked_keys:
                if key in outputs:
                    metric_totals[key] = metric_totals.get(key, 0.0) + (
                        float(outputs[key].item()) * batch_size
                    )

    total_examples = max(total_examples, 1)
    metrics = {"energy_objective_mean": total_loss / total_examples}
    for key, value in metric_totals.items():
        metrics[key] = value / total_examples
    return metrics


def run_structural_validation(
    generator,
    record: tuple[str, str],
    *,
    max_length: int,
    steps: int,
) -> dict[str, Any]:
    """Runs one structural sanity check on a real FASTA sequence.

    Args:
        generator: Configured ``QDiffusion`` instance.
        record: Input ``(header, sequence)`` pair.
        max_length: Optional truncation length before tokenization.
        steps: Number of decode steps for the sanity-check generation.

    Returns:
        dict[str, Any]: Structured summary of objective and generation behavior.
    """
    header, sequence = record
    target_tokens = encode_sequence(generator, sequence, max_length=max_length)
    with torch.no_grad():
        objective_outputs = generator.objective({"targets": target_tokens})
        generated_tokens = generator.generate(target_tokens, max_steps=steps)

    decoded_target = generator.tokenizer.batch_decode(
        target_tokens, skip_special_tokens=True
    )[0]
    decoded_output = generator.tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]
    generated_sequence = normalize_decoded_sequence(decoded_output)

    return {
        "header": header,
        "raw_length": len(sequence),
        "token_length": int(target_tokens.shape[1]),
        "objective_summary": summarize_objective(objective_outputs),
        "decoded_target_prefix": decoded_target[:80],
        "generated_prefix": generated_sequence[:80],
        "generated_length": len(generated_sequence),
    }


def run_generation_over_records(
    generator,
    records: list[tuple[str, str]],
    *,
    max_steps: int,
    seed_base: int,
    output_dir: Path,
    label: str,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Generates one sequence per input record and writes FASTA/TSV artifacts.

    Args:
        generator: Configured ``QDiffusion`` instance.
        records: Input FASTA records.
        max_steps: Decode step count for each sequence.
        seed_base: Base random seed; each record offsets from this value.
        output_dir: Directory for FASTA/TSV outputs.
        label: Artifact prefix label.

    Returns:
        tuple[list[tuple[str, str]], list[dict[str, Any]]]: A tuple of generated FASTA records and per-sequence summary rows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_records: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []

    for index, (header, sequence) in enumerate(records, start=1):
        target_tokens = encode_sequence(generator, sequence, max_length=None)
        seed = seed_base + index
        seed_torch(seed)

        with torch.no_grad():
            objective_outputs = generator.objective({"targets": target_tokens})
            generated_tokens = generator.generate(target_tokens, max_steps=max_steps)

        decoded_output = generator.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        generated_sequence = normalize_decoded_sequence(decoded_output)
        generated_records.append((header, generated_sequence))

        rows.append(
            {
                "index": index,
                "header": header,
                "label": label,
                "reference_length": len(sequence),
                "generated_length": len(generated_sequence),
                "energy_objective_mean": round(
                    float(objective_outputs["energy_objective"].mean().item()), 4
                ),
            }
        )

    write_fasta_records(
        output_dir / f"{label}_generated_sequences.fasta", generated_records
    )
    write_tsv_rows(output_dir / f"{label}_generation_summary.tsv", rows)

    return generated_records, rows


def write_markdown_report(
    path: Path,
    *,
    config,
    validation_summary: dict[str, Any],
    train_history: list[dict[str, Any]],
    baseline_quality: QualitySummary,
    guided_quality: QualitySummary,
    comparison_summary: dict[str, Any],
    selected_count: int,
    train_count: int,
    val_count: int,
    test_count: int,
    best_checkpoint_path: Path | None,
) -> None:
    """Writes the final markdown report for one full run."""
    lines = [
        "# Full Example Report",
        "",
        "## 1. Data Split",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| selected_records | {selected_count} |",
        f"| train_records | {train_count} |",
        f"| val_records | {val_count} |",
        f"| test_records | {test_count} |",
        "",
        "## 2. Structural Validation",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| header | {validation_summary['header']} |",
        f"| raw_length | {validation_summary['raw_length']} |",
        f"| token_length | {validation_summary['token_length']} |",
        f"| generated_length | {validation_summary['generated_length']} |",
        f"| objective_summary | {validation_summary['objective_summary']} |",
        "",
        "## 3. Training History",
        "",
        "| Epoch | Train energy_objective_mean | Val energy_objective_mean |",
        "|---:|---:|---:|",
    ]
    for row in train_history:
        lines.append(
            f"| {row['epoch']} | {row['train_energy_objective_mean']:.5f} | "
            f"{row['val_energy_objective_mean']:.5f} |"
        )
    lines += [
        "",
        "## 4. Baseline vs Guided Quality",
        "",
        "| Metric | Baseline | Guided |",
        "|---|---:|---:|",
        f"| amino_acid_jsd | {baseline_quality.amino_acid_jsd:.5f} | {guided_quality.amino_acid_jsd:.5f} |",
        f"| kmer2_jsd | {baseline_quality.kmer2_jsd:.5f} | {guided_quality.kmer2_jsd:.5f} |",
        f"| kmer3_jsd | {baseline_quality.kmer3_jsd:.5f} | {guided_quality.kmer3_jsd:.5f} |",
        f"| length_match_ratio | {baseline_quality.length_match_ratio:.5f} | {guided_quality.length_match_ratio:.5f} |",
        f"| identity_to_reference_mean | {baseline_quality.identity_to_reference_mean:.5f} | {guided_quality.identity_to_reference_mean:.5f} |",
        f"| uniqueness_ratio | {baseline_quality.uniqueness_ratio:.5f} | {guided_quality.uniqueness_ratio:.5f} |",
        f"| repeat_ratio_ge4 | {baseline_quality.repeat_ratio_ge4:.5f} | {guided_quality.repeat_ratio_ge4:.5f} |",
        "",
        "## 5. Baseline vs Guided Difference",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| changed_sequences | {comparison_summary['changed_sequences']} |",
        f"| changed_fraction | {comparison_summary['changed_fraction']:.5f} |",
        f"| baseline_identity_mean | {comparison_summary['baseline_identity_mean']:.5f} |",
        f"| guided_identity_mean | {comparison_summary['guided_identity_mean']:.5f} |",
        "",
        "## 6. Checkpoint",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| best_checkpoint | {best_checkpoint_path if best_checkpoint_path is not None else 'N/A'} |",
        f"| train_num_candidates | {config.train.num_candidates} |",
        f"| guided_num_candidates | {config.generate.num_candidates} |",
        f"| generation_steps | {config.generate.steps} |",
        "",
    ]
    save_markdown(path, lines)


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
    train_generator.train()
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
