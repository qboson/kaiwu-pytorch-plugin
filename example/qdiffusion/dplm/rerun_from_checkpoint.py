# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint rerun workflow for guided ``QDiffusion`` experiments."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from _example_bootstrap import ensure_repo_src_on_path
import torch

ensure_repo_src_on_path()

from dplm_builder import build_dplm_qdiffusion

os.environ.setdefault("BYPROT_EAGER_IMPORTS", "0")

CASE_ROOT = Path(__file__).resolve().parents[1]


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
        list[tuple[str, str]]: A list of ``(header, sequence)`` pairs.
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
        str: Normalized sequence text without spaces.
    """
    return sequence.replace(" ", "").strip()


def encode_sequence(generator, sequence: str, max_length: int | None = None) -> torch.Tensor:
    """Tokenizes one protein sequence to ``[1, seq_len]``.

    Args:
        generator: Configured ``QDiffusion`` instance.
        sequence: Protein sequence text.
        max_length: Optional truncation length before tokenization.

    Returns:
        torch.Tensor: Token tensor on the generator device.
    """
    if max_length is not None:
        sequence = sequence[:max_length]
    encoded = generator.tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return encoded["input_ids"].to(generator.device)


def sequence_identity(reference: str, candidate: str) -> float:
    """Computes position-wise identity over the shared prefix.

    Args:
        reference: Reference sequence text.
        candidate: Candidate sequence text.

    Returns:
        float: Shared-prefix identity ratio.
    """
    if not reference or not candidate:
        return 0.0
    compare_length = min(len(reference), len(candidate))
    if compare_length == 0:
        return 0.0
    matches = sum(1 for index in range(compare_length) if reference[index] == candidate[index])
    return matches / compare_length


def token_distribution(sequences: Iterable[str]) -> dict[str, int]:
    """Counts amino-acid frequencies across a sequence set.

    Args:
        sequences: Sequence collection to summarize.

    Returns:
        dict[str, int]: Per-token count dictionary.
    """
    counts: dict[str, int] = {}
    for sequence in sequences:
        for token in sequence:
            counts[token] = counts.get(token, 0) + 1
    return counts


def kmer_distribution(sequences: Iterable[str], k: int) -> dict[str, int]:
    """Counts k-mers across a sequence set.

    Args:
        sequences: Sequence collection to summarize.
        k: K-mer size.

    Returns:
        dict[str, int]: Per-k-mer count dictionary.
    """
    counts: dict[str, int] = {}
    for sequence in sequences:
        if len(sequence) < k:
            continue
        for index in range(len(sequence) - k + 1):
            kmer = sequence[index : index + k]
            counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def jsd_from_counts(reference_counts: dict[str, int], candidate_counts: dict[str, int]) -> float:
    """Computes Jensen-Shannon divergence from two count dictionaries.

    Args:
        reference_counts: Reference frequency counts.
        candidate_counts: Candidate frequency counts.

    Returns:
        float: Jensen-Shannon divergence value.
    """
    vocabulary = sorted(set(reference_counts) | set(candidate_counts))
    if not vocabulary:
        return 0.0
    reference_total = sum(reference_counts.values())
    candidate_total = sum(candidate_counts.values())
    if reference_total == 0 or candidate_total == 0:
        return 0.0

    reference_probs = [reference_counts.get(token, 0) / reference_total for token in vocabulary]
    candidate_probs = [candidate_counts.get(token, 0) / candidate_total for token in vocabulary]
    mean_probs = [(p + q) / 2.0 for p, q in zip(reference_probs, candidate_probs)]

    def kl_divergence(probs_a: list[float], probs_b: list[float]) -> float:
        value = 0.0
        for prob_a, prob_b in zip(probs_a, probs_b):
            if prob_a <= 0.0 or prob_b <= 0.0:
                continue
            value += prob_a * math.log(prob_a / prob_b)
        return value

    return 0.5 * kl_divergence(reference_probs, mean_probs) + 0.5 * kl_divergence(candidate_probs, mean_probs)


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
    flagged = sum(1 for sequence in sequences if max_repeat_run(sequence) >= min_repeat_run)
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
    reference_sequences = [normalize_decoded_sequence(sequence) for _, sequence in reference_records]
    generated_sequences = [normalize_decoded_sequence(sequence) for _, sequence in generated_records]

    paired_count = min(len(reference_sequences), len(generated_sequences))
    if paired_count == 0:
        raise ValueError("Need at least one aligned reference/generated pair for evaluation.")

    paired_references = reference_sequences[:paired_count]
    paired_generated = generated_sequences[:paired_count]
    length_match_ratio = sum(
        1 for reference, generated in zip(paired_references, paired_generated) if len(reference) == len(generated)
    ) / paired_count
    identity_to_reference_mean = sum(
        sequence_identity(reference, generated) for reference, generated in zip(paired_references, paired_generated)
    ) / paired_count

    return QualitySummary(
        label=label,
        reference_count=len(reference_sequences),
        generated_count=len(generated_sequences),
        mean_reference_length=sum(len(sequence) for sequence in reference_sequences) / len(reference_sequences),
        mean_generated_length=sum(len(sequence) for sequence in generated_sequences) / len(generated_sequences),
        length_match_ratio=length_match_ratio,
        identity_to_reference_mean=identity_to_reference_mean,
        amino_acid_jsd=jsd_from_counts(token_distribution(reference_sequences), token_distribution(generated_sequences)),
        kmer2_jsd=jsd_from_counts(kmer_distribution(reference_sequences, 2), kmer_distribution(generated_sequences, 2)),
        kmer3_jsd=jsd_from_counts(kmer_distribution(reference_sequences, 3), kmer_distribution(generated_sequences, 3)),
        uniqueness_ratio=uniqueness_ratio(generated_sequences),
        repeat_ratio_ge4=repeat_ratio(generated_sequences, min_repeat_run=min_repeat_run),
    )


def save_quality_summary(output_dir: Path, summary: QualitySummary) -> None:
    """Writes JSON and TSV summaries for one quality result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    with (output_dir / "quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with (output_dir / "quality_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(payload)


def load_trained_energy_weights(generator, checkpoint_path: str, device: str) -> None:
    """Loads a compact energy-side checkpoint into one generator."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    generator.energy_model.encoder.backbone.load_state_dict(state_dict["energy_encoder"])
    generator.energy_model.feature_projector.load_state_dict(
        state_dict["feature_projector"]
    )
    generator.energy_model.energy_rbm.load_state_dict(state_dict["energy_rbm"])
    generator.energy_head.load_state_dict(state_dict["energy_head"])
    generator.vocab_proj.load_state_dict(state_dict["vocab_proj"])


def run_generation_over_records(
    generator,
    records: list[tuple[str, str]],
    *,
    max_steps: int,
    seed_base: int,
    output_dir: Path,
    label: str,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Generates one sequence per input record and writes FASTA/TSV artifacts."""
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
            generated_tokens = generator.generate(target_tokens, max_steps=max_steps)

        decoded_output = generator.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        generated_sequence = normalize_decoded_sequence(decoded_output)
        generated_records.append((header, generated_sequence))
        rows.append(
            {
                "index": index,
                "header": header,
                "label": label,
                "reference_length": len(sequence),
                "generated_length": len(generated_sequence),
                "identity_to_reference": round(sequence_identity(sequence, generated_sequence), 4),
            }
        )

    write_fasta_records(output_dir / f"{label}_generated_sequences.fasta", generated_records)
    with (output_dir / f"{label}_generation_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    return generated_records, rows


def compare_generation_sets(
    reference_records: list[tuple[str, str]],
    baseline_records: list[tuple[str, str]],
    guided_records: list[tuple[str, str]],
) -> dict[str, Any]:
    """Compares baseline and guided generations against each other."""
    paired_count = min(len(reference_records), len(baseline_records), len(guided_records))
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


def default_outputs_root() -> Path:
    """Returns the outputs root for the final clean case directory."""
    return CASE_ROOT / "outputs"


def find_latest_real_full_run() -> str:
    """Finds the newest real full-workflow output directory."""
    candidates = sorted(
        default_outputs_root().glob("real_full_workflow_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No real_full_workflow_* output directory was found. "
            "Run full training first or edit GuidedRerunConfig.previous_run_dir."
        )
    return str(candidates[0])


def find_best_checkpoint(run_dir: Path) -> str:
    """Finds the newest best_epoch checkpoint inside one workflow output."""
    candidates = sorted(
        (run_dir / "checkpoints").glob("best_epoch_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No best_epoch_*.pt checkpoint was found under {run_dir / 'checkpoints'}.")
    return str(candidates[0])


@dataclass
class GuidedRerunConfig:
    """Config for rerunning guided generation from an existing checkpoint."""

    previous_run_dir: str
    best_checkpoint: str | None = None
    guided_num_candidates: int = 8
    guided_energy_temperature: float = 1.25
    guided_proposal_temperature: float = 0.3
    guided_proposal_noise_scale: float = 1.0
    guided_disable_resample: bool = False
    guided_resample_ratio: float = 0.20
    guided_resample_top_p: float = 0.90
    generation_steps: int | None = None
    freeze_proposal: bool | None = None
    seed: int | None = None
    label: str = "guided_rerun"


def write_rerun_report(
    path: Path,
    *,
    config: GuidedRerunConfig,
    source_config: dict[str, Any],
    baseline_quality: QualitySummary,
    guided_quality: QualitySummary,
    comparison_summary: dict[str, Any],
    checkpoint_path: str,
) -> None:
    """Writes a compact markdown report for one guided rerun."""
    baseline_metrics = asdict(baseline_quality)
    guided_metrics = asdict(guided_quality)
    lines = [
        "# Guided Rerun Report",
        "",
        "## 1. Source Workflow",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| previous_run_dir | {config.previous_run_dir} |",
        f"| best_checkpoint | {checkpoint_path} |",
        "",
        "## 2. Guided Parameters",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| guided_num_candidates | {config.guided_num_candidates} |",
        f"| guided_energy_temperature | {config.guided_energy_temperature} |",
        f"| guided_proposal_temperature | {config.guided_proposal_temperature} |",
        f"| guided_proposal_noise_scale | {config.guided_proposal_noise_scale} |",
        f"| guided_disable_resample | {int(config.guided_disable_resample)} |",
        f"| guided_resample_ratio | {config.guided_resample_ratio:.4f} |",
        f"| guided_resample_top_p | {config.guided_resample_top_p:.4f} |",
        f"| generation_steps | {config.generation_steps if config.generation_steps is not None else source_config['generation_steps']} |",
        "",
        "## 3. Baseline vs Guided Quality",
        "",
        "| Metric | Baseline | Guided |",
        "|---|---:|---:|",
        f"| amino_acid_jsd | {baseline_metrics['amino_acid_jsd']:.5f} | {guided_metrics['amino_acid_jsd']:.5f} |",
        f"| kmer2_jsd | {baseline_metrics['kmer2_jsd']:.5f} | {guided_metrics['kmer2_jsd']:.5f} |",
        f"| kmer3_jsd | {baseline_metrics['kmer3_jsd']:.5f} | {guided_metrics['kmer3_jsd']:.5f} |",
        f"| length_match_ratio | {baseline_metrics['length_match_ratio']:.5f} | {guided_metrics['length_match_ratio']:.5f} |",
        f"| identity_to_reference_mean | {baseline_metrics['identity_to_reference_mean']:.5f} | {guided_metrics['identity_to_reference_mean']:.5f} |",
        f"| uniqueness_ratio | {baseline_metrics['uniqueness_ratio']:.5f} | {guided_metrics['uniqueness_ratio']:.5f} |",
        f"| repeat_ratio_ge4 | {baseline_metrics['repeat_ratio_ge4']:.5f} | {guided_metrics['repeat_ratio_ge4']:.5f} |",
        "",
        "## 4. Baseline vs Guided Difference",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| changed_sequences | {comparison_summary['changed_sequences']} |",
        f"| changed_fraction | {comparison_summary['changed_fraction']:.5f} |",
        f"| baseline_identity_mean | {comparison_summary['baseline_identity_mean']:.5f} |",
        f"| guided_identity_mean | {comparison_summary['guided_identity_mean']:.5f} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Reruns guided generation/evaluation from an existing best checkpoint."""
    config = GuidedRerunConfig(previous_run_dir=find_latest_real_full_run())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    previous_run_dir = Path(config.previous_run_dir)
    source_config = json.loads((previous_run_dir / "run_config.json").read_text(encoding="utf-8"))
    checkpoint_path = config.best_checkpoint or find_best_checkpoint(previous_run_dir)
    output_dir = previous_run_dir / f"{config.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Previous workflow: {previous_run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    test_records = read_fasta_records(previous_run_dir / "data_splits" / "test.fasta")
    baseline_records = read_fasta_records(previous_run_dir / "baseline" / "proposal_only_generated_sequences.fasta")

    generator = (
        build_dplm_qdiffusion(
            proposal_ckpt=source_config["proposal_ckpt"],
            energy_ckpt=source_config["energy_ckpt"],
            num_candidates=config.guided_num_candidates,
            proposal_temperature=config.guided_proposal_temperature,
            proposal_noise_scale=config.guided_proposal_noise_scale,
            energy_temperature=config.guided_energy_temperature,
            disable_resample=config.guided_disable_resample,
            resample_ratio=config.guided_resample_ratio,
            resample_top_p=config.guided_resample_top_p,
            freeze_proposal=(
                source_config["freeze_proposal"]
                if config.freeze_proposal is None
                else config.freeze_proposal
            ),
        )
        .eval()
        .to(device)
    )
    load_trained_energy_weights(generator, checkpoint_path, device)
    guided_records, _ = run_generation_over_records(
        generator,
        test_records,
        max_steps=(source_config["generation_steps"] if config.generation_steps is None else config.generation_steps),
        seed_base=(source_config["seed"] if config.seed is None else config.seed),
        output_dir=output_dir,
        label=config.label,
    )

    baseline_quality = evaluate_generation_quality(test_records, baseline_records, label="proposal_only")
    guided_quality = evaluate_generation_quality(test_records, guided_records, label=config.label)
    save_quality_summary(output_dir / "baseline_eval", baseline_quality)
    save_quality_summary(output_dir / "guided_eval", guided_quality)

    comparison_summary = compare_generation_sets(test_records, baseline_records, guided_records)
    save_json(output_dir / "rerun_config.json", asdict(config))
    save_json(output_dir / "baseline_vs_guided.json", comparison_summary)
    write_rerun_report(
        output_dir / "REPORT.md",
        config=config,
        source_config=source_config,
        baseline_quality=baseline_quality,
        guided_quality=guided_quality,
        comparison_summary=comparison_summary,
        checkpoint_path=checkpoint_path,
    )

    print("Guided rerun completed.")
    print(f"Report: {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
