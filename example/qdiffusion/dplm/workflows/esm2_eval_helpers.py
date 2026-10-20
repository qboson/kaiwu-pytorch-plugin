"""Helper routines for ESM2 distance evaluation workflows."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from ..utils.io import (
    normalize_sequence,
    save_markdown,
    write_fasta_records,
)
from ..utils.runtime import seed_torch


@dataclass
class PairDistanceRow:
    """Per-pair embedding-distance result."""

    index: int
    reference_header: str
    candidate_header: str
    reference_length: int
    candidate_length: int
    cosine_distance: float
    l2_distance: float


@dataclass
class DistanceSummary:
    """Aggregate metrics for one generated FASTA against one reference FASTA."""

    label: str
    paired_count: int
    mean_reference_length: float
    mean_candidate_length: float
    mean_cosine_distance: float
    median_cosine_distance: float
    min_cosine_distance: float
    max_cosine_distance: float
    mean_l2_distance: float
    median_l2_distance: float
    min_l2_distance: float
    max_l2_distance: float


def build_full_mask_input(generator, sequence_length: int) -> torch.Tensor:
    """Builds an all-mask input of one sequence length.

    Args:
        generator: Assembled ``QDiffusion`` whose tokenizer/device are used.

        sequence_length: Number of masked positions, matching the reference
            sequence so the generated output keeps its length.

    Returns:
        torch.Tensor: Token ids shaped ``[1, sequence_length + specials]``
        on the generator device.
    """
    masked_sequence = "".join(["<mask>"] * sequence_length)
    encoded = generator.tokenizer.batch_encode_plus(
        [masked_sequence],
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    return encoded["input_ids"].to(generator.device)


def run_masked_generation_over_records(
    generator,
    records: list[tuple[str, str]],
    *,
    max_steps: int,
    seed_base: int,
    output_fasta_path: Path,
    label: str,
) -> list[tuple[str, str]]:
    """Generates from all-mask inputs so each sequence keeps its reference length.

    Deliberately distinct from ``workflow_helpers.run_generation_over_records``,
    which regenerates while conditioned on the reference sequence itself.
    """
    generated_records: list[tuple[str, str]] = []
    print(
        f"[{label}] starting generation for {len(records)} sequences "
        f"with max_steps={max_steps}"
    )

    for index, (header, sequence) in enumerate(records, start=1):
        sequence = normalize_sequence(sequence)
        input_tokens = build_full_mask_input(generator, len(sequence))
        partial_masks = input_tokens.ne(generator.mask_id)
        seed_torch(seed_base + index)

        with torch.no_grad():
            generated_tokens = generator.generate(
                input_tokens,
                max_steps=max_steps,
                partial_masks=partial_masks,
            )

        decoded = generator.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        generated_records.append((header, normalize_sequence(decoded)))
        print(f"[{label}] finished sequence {index}/{len(records)}: {header}")

    write_fasta_records(output_fasta_path, generated_records)
    print(f"[{label}] generated FASTA: {output_fasta_path}")
    return generated_records


def pair_records_by_order(
    reference_records: list[tuple[str, str]],
    candidate_records: list[tuple[str, str]],
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Pairs reference and candidate records position by position.

    Args:
        reference_records: Reference FASTA records.

        candidate_records: Generated FASTA records.

    Returns:
        list: ``(reference, candidate)`` pairs truncated to the shorter of
        the two lists.
    """
    paired_count = min(len(reference_records), len(candidate_records))
    return list(zip(reference_records[:paired_count], candidate_records[:paired_count]))


def chunked(
    items: list[tuple[str, str]], batch_size: int
) -> Iterable[list[tuple[str, str]]]:
    """Yields fixed-size slices of records for batched embedding.

    Args:
        items: FASTA records.

        batch_size: Slice size; the final slice may be smaller.

    Yields:
        list[tuple[str, str]]: One batch of records.
    """
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_esm2_model(
    model_name: str, device: torch.device
) -> tuple[torch.nn.Module, object]:
    """Loads one frozen ESM2 model from ``esm.pretrained``.

    Args:
        model_name: Loader name such as ``esm2_t33_650M_UR50D``.

        device: Device the model is moved to.

    Returns:
        tuple: The eval-mode model and its alphabet.

    Raises:
        ImportError: If the optional ``esm`` package is missing.

        ValueError: If the loader name is not an ESM2 entry point.
    """
    print(f"Loading ESM2 model: {model_name} on {device}")
    try:
        import esm
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "Missing dependency 'esm'. Install facebookresearch/esm first "
            "(see example/qdiffusion/requirements.txt), then rerun."
        ) from exc

    if not hasattr(esm.pretrained, model_name):
        available = sorted(
            name for name in dir(esm.pretrained) if name.startswith("esm2_")
        )
        raise ValueError(
            f"Unsupported esm.pretrained model '{model_name}'. "
            f"Available examples: {', '.join(available[:8])}"
        )

    loader = getattr(esm.pretrained, model_name)
    model, alphabet = loader()
    model = model.eval().to(device)
    return model, alphabet


def embed_sequences(
    records: list[tuple[str, str]],
    *,
    model: torch.nn.Module,
    alphabet: object,
    device: torch.device,
    batch_size: int,
    pooling: str,
) -> dict[str, torch.Tensor]:
    """Embeds FASTA records with ESM2 and pools per-sequence vectors.

    Args:
        records: ``(header, sequence)`` FASTA records.

        model: Frozen ESM2 model from ``load_esm2_model``.

        alphabet: Matching ESM alphabet and batch converter.

        device: Device used for embedding.

        batch_size: Records embedded per forward pass.

        pooling: One of ``mean``, ``cls``/``bos``, or ``eos``.

    Returns:
        dict[str, torch.Tensor]: Header-keyed pooled embeddings on CPU.

    Raises:
        ValueError: If a sequence is empty or the pooling mode is unknown.
    """
    if not records:
        return {}

    print(
        f"Embedding {len(records)} sequences with batch_size={batch_size} "
        f"and pooling={pooling}"
    )
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers
    embeddings: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for batch_records in chunked(records, batch_size):
            _, _, tokens = batch_converter(batch_records)
            tokens = tokens.to(device)

            outputs = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = outputs["representations"][repr_layer]
            lengths = (tokens != alphabet.padding_idx).sum(dim=1)

            for row_index, (header, sequence) in enumerate(batch_records):
                seq_len = len(sequence)
                if seq_len == 0:
                    raise ValueError(
                        f"Encountered empty sequence for header '{header}'."
                    )

                residue_repr = token_representations[row_index, 1 : seq_len + 1]
                if pooling == "mean":
                    pooled = residue_repr.mean(dim=0)
                elif pooling in {"cls", "bos"}:
                    pooled = token_representations[row_index, 0]
                elif pooling == "eos":
                    eos_index = int(lengths[row_index].item()) - 1
                    pooled = token_representations[row_index, eos_index]
                else:
                    raise ValueError(f"Unsupported pooling mode: {pooling}")

                embeddings[header] = pooled.detach().cpu()

    return embeddings


def summarize_distances(label: str, rows: list[PairDistanceRow]) -> DistanceSummary:
    """Aggregates per-pair distance rows into one summary.

    Args:
        label: Strategy label recorded in the summary.

        rows: Pair rows produced by ``evaluate_candidate_set``.

    Returns:
        DistanceSummary: Count, length, and distance statistics.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError(f"No aligned rows were available for label '{label}'.")

    cosine_distances = [row.cosine_distance for row in rows]
    l2_distances = [row.l2_distance for row in rows]
    reference_lengths = [row.reference_length for row in rows]
    candidate_lengths = [row.candidate_length for row in rows]

    return DistanceSummary(
        label=label,
        paired_count=len(rows),
        mean_reference_length=statistics.mean(reference_lengths),
        mean_candidate_length=statistics.mean(candidate_lengths),
        mean_cosine_distance=statistics.mean(cosine_distances),
        median_cosine_distance=statistics.median(cosine_distances),
        min_cosine_distance=min(cosine_distances),
        max_cosine_distance=max(cosine_distances),
        mean_l2_distance=statistics.mean(l2_distances),
        median_l2_distance=statistics.median(l2_distances),
        min_l2_distance=min(l2_distances),
        max_l2_distance=max(l2_distances),
    )


def evaluate_candidate_set(
    *,
    label: str,
    reference_records: list[tuple[str, str]],
    candidate_records: list[tuple[str, str]],
    reference_embeddings: dict[str, torch.Tensor],
    candidate_embeddings: dict[str, torch.Tensor],
) -> tuple[list[PairDistanceRow], DistanceSummary]:
    """Scores candidate records against the reference by embedding distance.

    Args:
        label: Strategy label attached to the rows and summary.

        reference_records: Reference FASTA records.

        candidate_records: Generated FASTA records.

        reference_embeddings: Header-keyed reference embeddings.

        candidate_embeddings: Header-keyed candidate embeddings.

    Returns:
        tuple: Per-pair ``PairDistanceRow`` rows and the aggregate
        ``DistanceSummary``.

    Raises:
        ValueError: If no aligned pairs exist.
    """
    pairs = pair_records_by_order(reference_records, candidate_records)

    if not pairs:
        raise ValueError(
            f"No aligned pairs found for '{label}'. "
            "Check the FASTA record counts and order."
        )

    rows: list[PairDistanceRow] = []
    for index, ((ref_header, ref_seq), (cand_header, cand_seq)) in enumerate(
        pairs, start=1
    ):
        ref_embedding = reference_embeddings[ref_header]
        cand_embedding = candidate_embeddings[cand_header]
        cosine_distance = (
            1.0
            - F.cosine_similarity(
                ref_embedding.unsqueeze(0), cand_embedding.unsqueeze(0), dim=-1
            ).item()
        )
        l2_distance = torch.norm(ref_embedding - cand_embedding, p=2).item()

        rows.append(
            PairDistanceRow(
                index=index,
                reference_header=ref_header,
                candidate_header=cand_header,
                reference_length=len(ref_seq),
                candidate_length=len(cand_seq),
                cosine_distance=float(cosine_distance),
                l2_distance=float(l2_distance),
            )
        )

    return rows, summarize_distances(label, rows)


def write_report(
    path: Path,
    *,
    reference_path: Path,
    baseline_path: Path | None,
    guided_path: Path | None,
    summaries: list[DistanceSummary],
    model_name: str,
    pooling: str,
) -> None:
    """Writes the Markdown ESM2 distance report.

    Args:
        path: Destination Markdown path.

        reference_path: Reference FASTA used for generation/evaluation.

        baseline_path: Baseline FASTA, if produced.

        guided_path: Guided FASTA, if produced.

        summaries: One ``DistanceSummary`` per evaluated strategy.

        model_name: ESM2 model name used for embeddings.

        pooling: Pooling mode used for embeddings.
    """
    lines = [
        "# ESM2 Embedding Distance Report",
        "",
        f"- reference: `{reference_path}`",
        (
            f"- baseline: `{baseline_path}`"
            if baseline_path is not None
            else "- baseline: not provided"
        ),
        (
            f"- guided: `{guided_path}`"
            if guided_path is not None
            else "- guided: not provided"
        ),
        f"- esm2 model: `{model_name}`",
        f"- pooling: `{pooling}`",
        "",
        "| label | pairs | mean cosine dist | median cosine dist | mean l2 dist | median l2 dist |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.label,
                    str(summary.paired_count),
                    f"{summary.mean_cosine_distance:.6f}",
                    f"{summary.median_cosine_distance:.6f}",
                    f"{summary.mean_l2_distance:.6f}",
                    f"{summary.median_l2_distance:.6f}",
                ]
            )
            + " |"
        )

    if len(summaries) == 2:
        baseline_summary, guided_summary = summaries
        delta_cosine = (
            guided_summary.mean_cosine_distance - baseline_summary.mean_cosine_distance
        )
        delta_l2 = guided_summary.mean_l2_distance - baseline_summary.mean_l2_distance
        lines.extend(
            [
                "",
                "## Delta",
                "",
                f"- guided minus baseline mean cosine distance: {delta_cosine:.6f}",
                f"- guided minus baseline mean l2 distance: {delta_l2:.6f}",
            ]
        )

    save_markdown(path, lines)
