# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Training workflow for running full-corpus ``QDiffusion`` experiments."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from _example_bootstrap import ensure_repo_src_on_path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ensure_repo_src_on_path()

from dplm_builder import build_dplm_qdiffusion

os.environ.setdefault("BYPROT_EAGER_IMPORTS", "0")

CASE_ROOT = Path(__file__).resolve().parents[1]


def default_fasta_path() -> Path:
    """Returns the bundled FASTA path for this clean case directory.

    Returns:
        Path to the bundled FASTA file used by the workflow.
    """
    return CASE_ROOT / "data" / "UP000005640_9606.fasta"


def save_json(path: Path, payload: Any) -> None:
    """Writes stable JSON to disk.

    Args:
        path: Output JSON path.
        payload: JSON-serializable payload to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_fasta_records(fasta_path: Path) -> list[tuple[str, str]]:
    """Reads all FASTA records from one file.

    Args:
        fasta_path: FASTA file to read.

    Returns:
        A list of ``(header, sequence)`` pairs.
    """
    records: list[tuple[str, str]] = []
    header = ""
    sequence_parts: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(sequence_parts)))
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)

    if header:
        records.append((header, "".join(sequence_parts)))
    return records


def write_fasta_records(path: Path, records: list[tuple[str, str]]) -> None:
    """Writes FASTA records to disk.

    Args:
        path: Output FASTA path.
        records: FASTA records to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, (header, sequence) in enumerate(records, start=1):
            handle.write(f">seq_{index} {header}\n")
            handle.write(sequence + "\n")


def normalize_decoded_sequence(sequence: str) -> str:
    """Converts token-decoded output into FASTA-friendly sequence text.

    Args:
        sequence: Raw decoded tokenizer output.

    Returns:
        Normalized sequence text without spaces.
    """
    return sequence.replace(" ", "").strip()


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
        Filtered FASTA records.
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
        A tuple ``(train_records, val_records, test_records)``.

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


def encode_sequence(
    generator, sequence: str, max_length: int | None = None
) -> torch.Tensor:
    """Tokenizes one protein sequence to ``[1, seq_len]``.

    Args:
        generator: Configured ``QDiffusion`` instance.
        sequence: Protein sequence text.
        max_length: Optional truncation length before tokenization.

    Returns:
        Token tensor on the generator device.
    """
    if max_length is not None:
        sequence = sequence[:max_length]
    encoded = generator.tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return encoded["input_ids"].to(generator.device)


def summarize_objective(outputs: dict[str, torch.Tensor]) -> str:
    """Formats the main structural tensors returned by ``objective()``.

    Args:
        outputs: Output dictionary returned by ``generator.objective(...)``.

    Returns:
        Human-readable one-line summary for logging.
    """
    logits = outputs["logits"]
    targets = outputs["targets"]
    loss_mask = outputs["loss_mask"]
    weight = outputs["weight"]
    objective_ebm = outputs["objective_ebm"]
    return (
        f"logits={tuple(logits.shape)}, "
        f"targets={tuple(targets.shape)}, "
        f"masked_positions={int(loss_mask.sum().item())}, "
        f"weight_mean={float(weight.mean().item()):.4f}, "
        f"objective_ebm_mean={float(objective_ebm.mean().item()):.4f}"
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
        A dataloader producing tokenized batches.
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
        Aggregated epoch metrics.
    """
    training = optimizer is not None
    set_training_mode(generator, training=training)

    total_loss = 0.0
    total_examples = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(data_loader, desc=description, unit="batch"):
            outputs = generator.objective({"targets": batch["targets"]})
            loss = outputs["objective_ebm"].mean()

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

    return {"objective_ebm_mean": total_loss / max(total_examples, 1)}


def summarize_trainable_parameters(generator) -> dict[str, int]:
    """Counts total and trainable parameters.

    Args:
        generator: Configured ``QDiffusion`` instance.

    Returns:
        Dictionary with total and trainable parameter counts.
    """
    total = 0
    trainable = 0
    for parameter in generator.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total_parameters": total, "trainable_parameters": trainable}


def save_checkpoint(
    output_dir: Path,
    name: str,
    *,
    generator,
    epoch: int,
    metric: float,
) -> Path:
    """Saves a compact checkpoint containing only energy-side weights.

    Args:
        output_dir: Checkpoint output directory.
        name: Checkpoint filename.
        generator: Configured ``QDiffusion`` instance.
        epoch: Epoch index associated with the checkpoint.
        metric: Validation metric stored with the checkpoint.

    Returns:
        Path to the written checkpoint file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / name
    torch.save(
        {
            "epoch": epoch,
            "metric": metric,
            "state_dict": {
                "energy_model": generator.energy_model.state_dict(),
                "energy_head": generator.energy_head.state_dict(),
                "vocab_proj": generator.vocab_proj.state_dict(),
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_trained_energy_weights(generator, checkpoint_path: str, device: str) -> None:
    """Loads a compact energy-side checkpoint into one generator.

    Args:
        generator: Configured ``QDiffusion`` instance.
        checkpoint_path: Path to the compact checkpoint file.
        device: Device string used for checkpoint loading.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    generator.energy_model.load_state_dict(state_dict["energy_model"])
    generator.energy_head.load_state_dict(state_dict["energy_head"])
    generator.vocab_proj.load_state_dict(state_dict["vocab_proj"])


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
        Structured summary of objective and generation behavior.
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
        A tuple of generated FASTA records and per-sequence summary rows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_records: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []

    for index, (header, sequence) in enumerate(records, start=1):
        target_tokens = encode_sequence(generator, sequence, max_length=None)
        seed = seed_base + index
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
                "objective_ebm_mean": round(
                    float(objective_outputs["objective_ebm"].mean().item()), 4
                ),
            }
        )

    write_fasta_records(
        output_dir / f"{label}_generated_sequences.fasta", generated_records
    )
    with (output_dir / f"{label}_generation_summary.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        if rows:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0].keys()), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(rows)

    return generated_records, rows


def sequence_identity(reference: str, candidate: str) -> float:
    """Computes position-wise identity over the shared prefix."""
    if not reference or not candidate:
        return 0.0
    compare_length = min(len(reference), len(candidate))
    if compare_length == 0:
        return 0.0
    matches = sum(
        1 for index in range(compare_length) if reference[index] == candidate[index]
    )
    return matches / compare_length


def token_distribution(sequences: Iterable[str]) -> dict[str, int]:
    """Counts amino-acid frequencies across a sequence set."""
    counts: dict[str, int] = {}
    for sequence in sequences:
        for token in sequence:
            counts[token] = counts.get(token, 0) + 1
    return counts


def kmer_distribution(sequences: Iterable[str], k: int) -> dict[str, int]:
    """Counts k-mers across a sequence set."""
    counts: dict[str, int] = {}
    for sequence in sequences:
        if len(sequence) < k:
            continue
        for index in range(len(sequence) - k + 1):
            kmer = sequence[index : index + k]
            counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def jsd_from_counts(
    reference_counts: dict[str, int], candidate_counts: dict[str, int]
) -> float:
    """Computes Jensen-Shannon divergence from two count dictionaries."""
    vocabulary = sorted(set(reference_counts) | set(candidate_counts))
    if not vocabulary:
        return 0.0

    reference_total = sum(reference_counts.values())
    candidate_total = sum(candidate_counts.values())
    if reference_total == 0 or candidate_total == 0:
        return 0.0

    reference_probs = [
        reference_counts.get(token, 0) / reference_total for token in vocabulary
    ]
    candidate_probs = [
        candidate_counts.get(token, 0) / candidate_total for token in vocabulary
    ]
    mean_probs = [(p + q) / 2.0 for p, q in zip(reference_probs, candidate_probs)]

    def kl_divergence(probs_a: list[float], probs_b: list[float]) -> float:
        value = 0.0
        for prob_a, prob_b in zip(probs_a, probs_b):
            if prob_a <= 0.0 or prob_b <= 0.0:
                continue
            value += prob_a * math.log(prob_a / prob_b)
        return value

    return 0.5 * kl_divergence(reference_probs, mean_probs) + 0.5 * kl_divergence(
        candidate_probs, mean_probs
    )


def uniqueness_ratio(sequences: list[str]) -> float:
    """Computes the fraction of unique generated full sequences."""
    if not sequences:
        return 0.0
    return len(set(sequences)) / len(sequences)


def max_repeat_run(sequence: str) -> int:
    """Returns the longest run of one repeated token inside a sequence."""
    if not sequence:
        return 0
    best = 1
    current = 1
    for index in range(1, len(sequence)):
        if sequence[index] == sequence[index - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def repeat_ratio(sequences: list[str], *, min_repeat_run: int) -> float:
    """Computes the fraction of sequences containing a long repeat run."""
    if not sequences:
        return 0.0
    flagged = sum(
        1 for sequence in sequences if max_repeat_run(sequence) >= min_repeat_run
    )
    return flagged / len(sequences)


@dataclass
class QualitySummary:
    """Compact quality summary for one generated set."""

    label: str
    reference_count: int
    generated_count: int
    mean_reference_length: float
    mean_generated_length: float
    length_match_ratio: float
    identity_to_reference_mean: float
    amino_acid_jsd: float
    kmer2_jsd: float
    kmer3_jsd: float
    uniqueness_ratio: float
    repeat_ratio_ge4: float


def evaluate_generation_quality(
    reference_records: list[tuple[str, str]],
    generated_records: list[tuple[str, str]],
    *,
    label: str,
    min_repeat_run: int = 4,
) -> QualitySummary:
    """Evaluates one generated set against aligned reference records."""
    reference_sequences = [
        normalize_decoded_sequence(sequence) for _, sequence in reference_records
    ]
    generated_sequences = [
        normalize_decoded_sequence(sequence) for _, sequence in generated_records
    ]

    paired_count = min(len(reference_sequences), len(generated_sequences))
    if paired_count == 0:
        raise ValueError(
            "Need at least one aligned reference/generated pair for evaluation."
        )

    paired_references = reference_sequences[:paired_count]
    paired_generated = generated_sequences[:paired_count]

    length_match_ratio = (
        sum(
            1
            for reference, generated in zip(paired_references, paired_generated)
            if len(reference) == len(generated)
        )
        / paired_count
    )
    identity_to_reference_mean = (
        sum(
            sequence_identity(reference, generated)
            for reference, generated in zip(paired_references, paired_generated)
        )
        / paired_count
    )

    return QualitySummary(
        label=label,
        reference_count=len(reference_sequences),
        generated_count=len(generated_sequences),
        mean_reference_length=sum(len(sequence) for sequence in reference_sequences)
        / len(reference_sequences),
        mean_generated_length=sum(len(sequence) for sequence in generated_sequences)
        / len(generated_sequences),
        length_match_ratio=length_match_ratio,
        identity_to_reference_mean=identity_to_reference_mean,
        amino_acid_jsd=jsd_from_counts(
            token_distribution(reference_sequences),
            token_distribution(generated_sequences),
        ),
        kmer2_jsd=jsd_from_counts(
            kmer_distribution(reference_sequences, 2),
            kmer_distribution(generated_sequences, 2),
        ),
        kmer3_jsd=jsd_from_counts(
            kmer_distribution(reference_sequences, 3),
            kmer_distribution(generated_sequences, 3),
        ),
        uniqueness_ratio=uniqueness_ratio(generated_sequences),
        repeat_ratio_ge4=repeat_ratio(
            generated_sequences, min_repeat_run=min_repeat_run
        ),
    )


def save_quality_summary(output_dir: Path, summary: QualitySummary) -> None:
    """Writes JSON and TSV summaries for one quality result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    with (output_dir / "quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with (output_dir / "quality_summary.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(payload)


def compare_generation_sets(
    reference_records: list[tuple[str, str]],
    baseline_records: list[tuple[str, str]],
    guided_records: list[tuple[str, str]],
) -> dict[str, Any]:
    """Compares baseline and guided generations against each other."""
    paired_count = min(
        len(reference_records), len(baseline_records), len(guided_records)
    )
    changed_count = 0
    baseline_identity_sum = 0.0
    guided_identity_sum = 0.0

    for (_, reference), (_, baseline), (_, guided) in zip(
        reference_records[:paired_count],
        baseline_records[:paired_count],
        guided_records[:paired_count],
    ):
        if baseline != guided:
            changed_count += 1
        baseline_identity_sum += sequence_identity(reference, baseline)
        guided_identity_sum += sequence_identity(reference, guided)

    return {
        "changed_sequences": changed_count,
        "changed_fraction": changed_count / max(paired_count, 1),
        "baseline_identity_mean": baseline_identity_sum / max(paired_count, 1),
        "guided_identity_mean": guided_identity_sum / max(paired_count, 1),
    }


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
        "| Epoch | Train objective_ebm_mean | Val objective_ebm_mean |",
        "|---:|---:|---:|",
    ]
    for row in train_history:
        lines.append(
            f"| {row['epoch']} | {row['train_objective_ebm_mean']:.5f} | "
            f"{row['val_objective_ebm_mean']:.5f} |"
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
        f"| train_num_candidates | {config.train_num_candidates} |",
        f"| guided_num_candidates | {config.guided_num_candidates} |",
        f"| generation_steps | {config.generation_steps} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class RealFullWorkflowConfig:
    """Top-level config for the real full-corpus example."""

    fasta_path: str
    proposal_ckpt: str
    energy_ckpt: str
    min_length: int = 50
    max_length: int = 256
    max_records: int | None = None
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42
    epochs: int = 20
    min_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    train_num_candidates: int = 4
    guided_num_candidates: int = 8
    guided_energy_temperature: float = 1.25
    guided_proposal_temperature: float = 0.3
    guided_proposal_noise_scale: float = 1.0
    guided_disable_resample: bool = False
    guided_resample_ratio: float = 0.20
    guided_resample_top_p: float = 0.90
    validation_steps: int = 3
    generation_steps: int = 5
    freeze_proposal: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1
    early_stop_patience: int = 4
    require_cuda: bool = True


def main() -> None:
    """Runs the final self-contained full workflow."""
    config = RealFullWorkflowConfig(
        fasta_path=str(default_fasta_path()),
        proposal_ckpt="airkingbd/dplm_150m",
        energy_ckpt="airkingbd/dplm_150m",
    )

    has_cuda = torch.cuda.is_available()
    if config.require_cuda and not has_cuda:
        raise RuntimeError("CUDA is required for this real full-corpus workflow.")
    device = "cuda" if has_cuda else "cpu"

    output_root = CASE_ROOT / "outputs"
    run_name = datetime.now().strftime("real_full_workflow_%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if has_cuda:
        torch.cuda.manual_seed_all(config.seed)

    all_records = read_fasta_records(Path(config.fasta_path))
    selected_records = select_records(
        all_records,
        min_length=config.min_length,
        max_length=config.max_length,
        max_records=config.max_records,
    )
    train_records, val_records, test_records = split_train_val_test(
        selected_records,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
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

    validation_generator = (
        build_dplm_qdiffusion(
            proposal_ckpt=config.proposal_ckpt,
            energy_ckpt=config.energy_ckpt,
            num_candidates=1,
            freeze_proposal=config.freeze_proposal,
        )
        .eval()
        .to(device)
    )
    validation_summary = run_structural_validation(
        validation_generator,
        test_records[0],
        max_length=config.max_length,
        steps=config.validation_steps,
    )
    save_json(output_dir / "validation_summary.json", validation_summary)

    train_generator = build_dplm_qdiffusion(
        proposal_ckpt=config.proposal_ckpt,
        energy_ckpt=config.energy_ckpt,
        num_candidates=config.train_num_candidates,
        freeze_proposal=config.freeze_proposal,
    ).to(device)
    save_json(
        output_dir / "parameter_summary.json",
        summarize_trainable_parameters(train_generator),
    )

    train_loader = build_data_loader_from_records(
        train_generator, train_records, batch_size=config.batch_size, shuffle=True
    )
    val_loader = build_data_loader_from_records(
        train_generator, val_records, batch_size=config.batch_size, shuffle=False
    )

    optimizer = AdamW(
        [
            parameter
            for parameter in train_generator.parameters()
            if parameter.requires_grad
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    checkpoints_dir = output_dir / "checkpoints"
    train_history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_checkpoint_path: Path | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        train_metrics = run_epoch(
            train_generator,
            train_loader,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
            description=f"train-{epoch}",
        )
        val_metrics = run_epoch(
            train_generator,
            val_loader,
            optimizer=None,
            grad_clip_norm=config.grad_clip_norm,
            description=f"val-{epoch}",
        )

        epoch_summary = {
            "epoch": epoch,
            "train_objective_ebm_mean": train_metrics["objective_ebm_mean"],
            "val_objective_ebm_mean": val_metrics["objective_ebm_mean"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        train_history.append(epoch_summary)
        save_json(output_dir / "history.json", train_history)
        print(json.dumps(epoch_summary, ensure_ascii=False))

        current_val = val_metrics["objective_ebm_mean"]
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
            epoch >= config.min_epochs
            and epochs_without_improvement >= config.early_stop_patience
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
        metric=train_history[-1]["val_objective_ebm_mean"],
    )
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")

    baseline_generator = (
        build_dplm_qdiffusion(
            proposal_ckpt=config.proposal_ckpt,
            energy_ckpt=config.energy_ckpt,
            num_candidates=1,
            freeze_proposal=config.freeze_proposal,
        )
        .eval()
        .to(device)
    )
    baseline_records, _ = run_generation_over_records(
        baseline_generator,
        test_records,
        max_steps=config.generation_steps,
        seed_base=config.seed,
        output_dir=output_dir / "baseline",
        label="proposal_only",
    )

    guided_generator = (
        build_dplm_qdiffusion(
            proposal_ckpt=config.proposal_ckpt,
            energy_ckpt=config.energy_ckpt,
            num_candidates=config.guided_num_candidates,
            proposal_temperature=config.guided_proposal_temperature,
            proposal_noise_scale=config.guided_proposal_noise_scale,
            energy_temperature=config.guided_energy_temperature,
            disable_resample=config.guided_disable_resample,
            resample_ratio=config.guided_resample_ratio,
            resample_top_p=config.guided_resample_top_p,
            freeze_proposal=config.freeze_proposal,
        )
        .eval()
        .to(device)
    )
    if best_checkpoint_path is None:
        raise RuntimeError("Training did not produce a best checkpoint.")
    load_trained_energy_weights(guided_generator, str(best_checkpoint_path), device)
    guided_records, _ = run_generation_over_records(
        guided_generator,
        test_records,
        max_steps=config.generation_steps,
        seed_base=config.seed,
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
