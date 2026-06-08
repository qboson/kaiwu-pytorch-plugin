# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

try:
    from ..utils.io import (
        normalize_decoded_sequence,
        save_markdown,
        write_fasta_records,
        write_tsv_rows,
    )
    from ..utils.metrics import QualitySummary
    from ..utils.runtime import encode_sequence, seed_torch
except ImportError:  # pragma: no cover - direct script-path compatibility
    from utils.io import (
        normalize_decoded_sequence,
        save_markdown,
        write_fasta_records,
        write_tsv_rows,
    )
    from utils.metrics import QualitySummary
    from utils.runtime import encode_sequence, seed_torch


def select_records(
    records: list[tuple[str, str]],
    *,
    min_length: int,
    max_length: int,
    max_records: int | None,
) -> list[tuple[str, str]]:
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
    training = optimizer is not None
    if training:
        generator.train()
        if getattr(generator, "proposal_model", None) is not None:
            generator.proposal_model.eval()
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


def run_structural_validation(
    generator,
    record: tuple[str, str],
    *,
    max_length: int,
    steps: int,
) -> dict[str, Any]:
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
