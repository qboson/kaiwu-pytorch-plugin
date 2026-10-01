# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""Small workflow helpers kept out of the reader-facing training script."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils.io import (
    normalize_decoded_sequence,
    save_markdown,
    write_fasta_records,
    write_tsv_rows,
)
from ..utils.metrics import QualitySummary
from ..utils.runtime import encode_sequence, seed_torch


def select_records(
    records: list[tuple[str, str]],
    *,
    min_length: int,
    max_length: int,
    max_records: int | None,
) -> list[tuple[str, str]]:
    """Selects records whose sequence length falls inside the bounds.

    Args:
        records: ``(header, sequence)`` FASTA records.

        min_length: Minimum allowed sequence length.

        max_length: Maximum allowed sequence length.

        max_records: Optional cap on the number of selected records.

    Returns:
        list[tuple[str, str]]: Records within bounds, in input order.
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
    """Splits records into deterministic train/val/test partitions.

    Args:
        records: ``(header, sequence)`` FASTA records.

        val_ratio: Fraction of records assigned to validation.

        test_ratio: Fraction of records assigned to test.

        seed: Shuffle seed keeping the split reproducible.

    Returns:
        tuple: Train, validation, and test record lists.

    Raises:
        ValueError: If fewer than three records are supplied or the
            val/test sizes would consume every record.
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


class FastaSequenceDataset(Dataset[dict[str, str]]):
    """Dataset over ``(header, sequence)`` FASTA records.

    Attributes:
        records: The ``(header, sequence)`` pairs served per index.
    """

    def __init__(self, records: list[tuple[str, str]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        header, sequence = self.records[index]
        return {"header": header, "sequence": sequence}


def build_data_loader_from_records(
    generator, records, *, batch_size: int, shuffle: bool
) -> DataLoader:
    """Builds a tokenizing DataLoader over FASTA records.

    Args:
        generator: Assembled ``QDiffusion`` whose tokenizer/device are used.

        records: ``(header, sequence)`` FASTA records.

        batch_size: Batch size forwarded to the DataLoader.

        shuffle: Whether to shuffle the record order.

    Returns:
        DataLoader: Batches of ``{"headers", "targets"}`` on the generator
        device.
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

    return DataLoader(
        FastaSequenceDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


def run_epoch(
    generator,
    data_loader: DataLoader,
    *,
    optimizer: AdamW | None,
    grad_clip_norm: float,
    description: str,
) -> dict[str, float]:
    """Runs one training or evaluation epoch over a DataLoader.

    Args:
        generator: Assembled ``QDiffusion`` to train or evaluate.

        data_loader: Batches of tokenized targets.

        optimizer: Optimizer for training epochs; ``None`` for evaluation.

        grad_clip_norm: Global-norm gradient clip; non-positive disables it.

        description: Progress-bar label.

    Returns:
        dict[str, float]: Example-weighted objective mean plus the tracked
        energy metrics.
    """
    training = optimizer is not None
    if training:
        # ``QDiffusion.train()`` keeps a frozen proposal model in eval mode.
        generator.train()
    else:
        generator.eval()

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

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(data_loader, desc=description, unit="batch"):
            outputs = generator.objective({"targets": batch["targets"]})
            loss = outputs["energy_objective"].mean()

            if training:
                optimizer.zero_grad(set_to_none=True)
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


def summarize_objective(outputs: dict[str, torch.Tensor]) -> str:
    """Renders one objective output dictionary as a compact log line."""
    logits = outputs["logits"]
    targets = outputs["targets"]
    loss_mask = outputs["loss_mask"]
    energy_objective = outputs["energy_objective"]
    zero_energy = torch.tensor(0.0)
    positive_energy = float(outputs.get("positive_energy_mean", zero_energy).item())
    negative_energy = float(outputs.get("negative_energy_mean", zero_energy).item())
    return (
        f"logits={tuple(logits.shape)}, "
        f"targets={tuple(targets.shape)}, "
        f"masked_positions={int(loss_mask.sum().item())}, "
        f"energy_objective_mean={float(energy_objective.mean().item()):.4f}, "
        f"positive_energy_mean={positive_energy:.4f}, "
        f"negative_energy_mean={negative_energy:.4f}"
    )


def run_structural_validation(
    generator,
    record: tuple[str, str],
    *,
    max_length: int,
    steps: int,
) -> dict[str, Any]:
    """Sanity-checks one record through both objective and generate.

    Args:
        generator: Assembled ``QDiffusion``.

        record: ``(header, sequence)`` FASTA record to probe.

        max_length: Maximum tokenization length.

        steps: Decode steps for the generation probe.

    Returns:
        dict[str, Any]: Lengths, decoded prefixes, and an objective summary.
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
    """Generates conditioned on each reference record and writes artifacts.

    Deliberately distinct from
    ``esm2_eval_helpers.run_masked_generation_over_records``: this variant
    conditions on the reference sequence itself instead of all-mask inputs.

    Args:
        generator: Assembled ``QDiffusion``.

        records: ``(header, sequence)`` reference FASTA records.

        max_steps: Decode steps per sequence.

        seed_base: Base seed; per-record seeds offset by the record index.

        output_dir: Directory receiving the generated FASTA and summary TSV.

        label: Filename prefix distinguishing baseline from guided runs.

    Returns:
        tuple: Generated ``(header, sequence)`` records and their per-record
        summary rows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_records: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []

    for index, (header, sequence) in enumerate(records, start=1):
        target_tokens = encode_sequence(generator, sequence, max_length=None)
        seed_torch(seed_base + index)

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
    """Writes the human-readable Markdown run report.

    Args:
        path: Destination Markdown path.

        config: Workflow configuration snapshot.

        validation_summary: Output of ``run_structural_validation``.

        train_history: Per-epoch metric dictionaries.

        baseline_quality: Quality metrics of the proposal-only run.

        guided_quality: Quality metrics of the guided run.

        comparison_summary: Baseline/guided difference summary.

        selected_count: Number of records after length filtering.

        train_count: Number of training records.

        val_count: Number of validation records.

        test_count: Number of test records.

        best_checkpoint_path: Checkpoint selected on validation, if any.
    """
    best_checkpoint_text = (
        str(best_checkpoint_path) if best_checkpoint_path is not None else "N/A"
    )
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
        f"| amino_acid_jsd | {baseline_quality.amino_acid_jsd:.5f} | "
        f"{guided_quality.amino_acid_jsd:.5f} |",
        f"| kmer2_jsd | {baseline_quality.kmer2_jsd:.5f} | "
        f"{guided_quality.kmer2_jsd:.5f} |",
        f"| kmer3_jsd | {baseline_quality.kmer3_jsd:.5f} | "
        f"{guided_quality.kmer3_jsd:.5f} |",
        f"| length_match_ratio | {baseline_quality.length_match_ratio:.5f} | "
        f"{guided_quality.length_match_ratio:.5f} |",
        f"| identity_to_reference_mean | "
        f"{baseline_quality.identity_to_reference_mean:.5f} | "
        f"{guided_quality.identity_to_reference_mean:.5f} |",
        f"| uniqueness_ratio | {baseline_quality.uniqueness_ratio:.5f} | "
        f"{guided_quality.uniqueness_ratio:.5f} |",
        f"| repeat_ratio_ge4 | {baseline_quality.repeat_ratio_ge4:.5f} | "
        f"{guided_quality.repeat_ratio_ge4:.5f} |",
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
        f"| best_checkpoint | {best_checkpoint_text} |",
        f"| train_num_candidates | {config.train.num_candidates} |",
        f"| guided_num_candidates | {config.generate.num_candidates} |",
        f"| generation_steps | {config.generate.steps} |",
        "",
    ]
    save_markdown(path, lines)
