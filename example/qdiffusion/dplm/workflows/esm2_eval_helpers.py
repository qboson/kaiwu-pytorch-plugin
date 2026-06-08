"""Helper routines for ESM2 distance evaluation workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

try:
    from ..utils.io import (
        normalize_sequence,
        save_json,
        save_markdown,
        write_csv_rows,
        write_fasta_records,
    )
    from ..utils.runtime import seed_torch
except ImportError:  # pragma: no cover - direct script-path compatibility
    from utils.io import (
        normalize_sequence,
        save_json,
        save_markdown,
        write_csv_rows,
        write_fasta_records,
    )
    from utils.runtime import seed_torch

try:
    import esm
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise SystemExit(
        "Missing dependency 'esm'. Install facebookresearch/esm first, then rerun."
    ) from exc


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


def maybe_limit_records(
    records: list[tuple[str, str]], max_records: int | None
) -> list[tuple[str, str]]:
    if max_records is None:
        return records
    return records[:max_records]


def build_full_mask_input(generator, sequence_length: int) -> torch.Tensor:
    masked_sequence = "".join(["<mask>"] * sequence_length)
    encoded = generator.tokenizer.batch_encode_plus(
        [masked_sequence],
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    return encoded["input_ids"].to(generator.device)


def run_generation_over_records(
    generator,
    records: list[tuple[str, str]],
    *,
    max_steps: int,
    seed_base: int,
    output_fasta_path: Path,
    label: str,
) -> list[tuple[str, str]]:
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
    paired_count = min(len(reference_records), len(candidate_records))
    return list(zip(reference_records[:paired_count], candidate_records[:paired_count]))


def pair_records_by_header(
    reference_records: list[tuple[str, str]],
    candidate_records: list[tuple[str, str]],
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    candidate_map = {header: (header, seq) for header, seq in candidate_records}
    pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for header, sequence in reference_records:
        candidate = candidate_map.get(header)
        if candidate is not None:
            pairs.append(((header, sequence), candidate))
    return pairs


def chunked(
    items: list[tuple[str, str]], batch_size: int
) -> Iterable[list[tuple[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_esm2_model(
    model_name: str, device: torch.device
) -> tuple[torch.nn.Module, object]:
    print(f"Loading ESM2 model: {model_name} on {device}")
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
            normalized_batch = [
                (header, normalize_sequence(sequence))
                for header, sequence in batch_records
            ]
            _, _, tokens = batch_converter(normalized_batch)
            tokens = tokens.to(device)

            outputs = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = outputs["representations"][repr_layer]
            lengths = (tokens != alphabet.padding_idx).sum(dim=1)

            for row_index, (header, sequence) in enumerate(normalized_batch):
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
    if not rows:
        raise ValueError(f"No aligned rows were available for label '{label}'.")

    cosine_values = torch.tensor(
        [row.cosine_distance for row in rows], dtype=torch.float64
    )
    l2_values = torch.tensor([row.l2_distance for row in rows], dtype=torch.float64)
    reference_lengths = torch.tensor(
        [row.reference_length for row in rows], dtype=torch.float64
    )
    candidate_lengths = torch.tensor(
        [row.candidate_length for row in rows], dtype=torch.float64
    )

    return DistanceSummary(
        label=label,
        paired_count=len(rows),
        mean_reference_length=float(reference_lengths.mean().item()),
        mean_candidate_length=float(candidate_lengths.mean().item()),
        mean_cosine_distance=float(cosine_values.mean().item()),
        median_cosine_distance=float(cosine_values.median().item()),
        min_cosine_distance=float(cosine_values.min().item()),
        max_cosine_distance=float(cosine_values.max().item()),
        mean_l2_distance=float(l2_values.mean().item()),
        median_l2_distance=float(l2_values.median().item()),
        min_l2_distance=float(l2_values.min().item()),
        max_l2_distance=float(l2_values.max().item()),
    )


def evaluate_candidate_set(
    *,
    label: str,
    reference_records: list[tuple[str, str]],
    candidate_records: list[tuple[str, str]],
    reference_embeddings: dict[str, torch.Tensor],
    candidate_embeddings: dict[str, torch.Tensor],
    pair_mode: str,
) -> tuple[list[PairDistanceRow], DistanceSummary]:
    if pair_mode == "header":
        pairs = pair_records_by_header(reference_records, candidate_records)
    elif pair_mode == "order":
        pairs = pair_records_by_order(reference_records, candidate_records)
    else:
        raise ValueError(f"Unsupported pair mode: {pair_mode}")

    if not pairs:
        raise ValueError(
            f"No aligned pairs found for '{label}'. "
            "Check --pair-mode and FASTA headers/order."
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
                reference_length=len(normalize_sequence(ref_seq)),
                candidate_length=len(normalize_sequence(cand_seq)),
                cosine_distance=float(cosine_distance),
                l2_distance=float(l2_distance),
            )
        )

    return rows, summarize_distances(label, rows)


def write_rows_csv(path: Path, rows: list[PairDistanceRow]) -> None:
    write_csv_rows(path, [asdict(row) for row in rows])


def write_summary_json(path: Path, summary: DistanceSummary) -> None:
    save_json(path, asdict(summary))


def write_report(
    path: Path,
    *,
    reference_path: Path,
    baseline_path: Path | None,
    guided_path: Path | None,
    summaries: list[DistanceSummary],
    model_name: str,
    pair_mode: str,
    pooling: str,
) -> None:
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
        f"- pair mode: `{pair_mode}`",
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
        lines.extend(
            [
                "",
                "## Delta",
                "",
                f"- guided minus baseline mean cosine distance: "
                f"{guided_summary.mean_cosine_distance - baseline_summary.mean_cosine_distance:.6f}",
                f"- guided minus baseline mean l2 distance: "
                f"{guided_summary.mean_l2_distance - baseline_summary.mean_l2_distance:.6f}",
            ]
        )

    save_markdown(path, lines)
