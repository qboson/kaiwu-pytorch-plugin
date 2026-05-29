"""Sequence-quality metrics for Q-Diffusion DPLM examples."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .io import normalize_decoded_sequence, save_json, write_tsv_rows


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
    save_json(output_dir / "quality_summary.json", payload)
    write_tsv_rows(output_dir / "quality_summary.tsv", [payload])


def compare_generation_sets(
    reference_records: list[tuple[str, str]],
    baseline_records: list[tuple[str, str]],
    guided_records: list[tuple[str, str]],
) -> dict[str, float]:
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
