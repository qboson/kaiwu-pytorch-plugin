"""Evaluate ``QDiffusion`` experiment outputs with ESM2 embedding distances.

This script follows one fixed workflow:

1. read the reference FASTA
2. generate baseline sequences from all-mask inputs
3. generate guided sequences from all-mask inputs
4. embed reference/baseline/guided sequences with ESM2
5. compare baseline/guided against the reference set

Edit the config block inside ``main()`` before running:

    python example/qdiffusion/qdiffusion_protein/eval_esm2_distances.py
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Iterable

try:
    from .._example_bootstrap import ensure_repo_src_on_path
except ImportError:  # pragma: no cover - direct script-path compatibility
    _WORKFLOW_DIR = Path(__file__).resolve().parent
    _CASE_DIR = _WORKFLOW_DIR.parent
    if str(_CASE_DIR) not in sys.path:
        sys.path.insert(0, str(_CASE_DIR))
    from _example_bootstrap import ensure_repo_src_on_path
import torch
import torch.nn.functional as F

ensure_repo_src_on_path()

try:
    from ..qdiffusion_protein_builder import build_qdiffusion_protein
    from ..utils.io import (
        write_csv_rows,
        default_fasta_path,
        default_outputs_root,
        normalize_sequence,
        read_fasta_records,
        save_json,
        save_markdown,
        write_fasta_records,
    )
    from ..utils.runtime import load_trained_energy_weights, seed_torch
except ImportError:  # pragma: no cover - direct script-path compatibility
    from qdiffusion_protein_builder import build_qdiffusion_protein
    from utils.io import (
        write_csv_rows,
        default_fasta_path,
        default_outputs_root,
        normalize_sequence,
        read_fasta_records,
        save_json,
        save_markdown,
        write_fasta_records,
    )
    from utils.runtime import load_trained_energy_weights, seed_torch

os.environ.setdefault("BYPROT_EAGER_IMPORTS", "0")

try:
    import esm
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise SystemExit(
        "Missing dependency 'esm'. Install facebookresearch/esm first, then rerun."
    ) from exc

# ---------------------------------------------------------------------------
# Data containers
# These dataclasses are just structured outputs for generation/evaluation.
# ---------------------------------------------------------------------------
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


@dataclass
class GenerationConfig:
    """Generation settings used when this script also produces FASTA files."""

    proposal_ckpt: str
    energy_ckpt: str
    guided_checkpoint: str | None
    generation_steps: int
    seed: int
    freeze_proposal: bool
    guided_num_candidates: int
    guided_proposal_temperature: float
    guided_proposal_noise_scale: float
    guided_energy_temperature: float
    guided_disable_resample: bool
    guided_resample_ratio: float
    guided_resample_top_p: float
    bm_sampler_type: str
    bm_sampler_kwargs: dict[str, object] | None


@dataclass
class EvalConfig:
    """Top-level config edited directly in ``main()`` for local/server runs."""

    reference_fasta: Path
    proposal_ckpt: str
    energy_ckpt: str
    guided_checkpoint: str
    output_dir: Path
    device: str
    esm2_model: str
    pair_mode: str
    pooling: str
    batch_size: int
    max_records: int | None
    generation_steps: int
    seed: int
    freeze_proposal: bool
    guided_num_candidates: int
    guided_proposal_temperature: float
    guided_proposal_noise_scale: float
    guided_energy_temperature: float
    guided_disable_resample: bool
    guided_resample_ratio: float
    guided_resample_top_p: float
    bm_sampler_type: str
    bm_sampler_kwargs: dict[str, object] | None


def build_bm_kwargs(config: EvalConfig) -> dict[str, object]:
    """Builds shared BM reranker kwargs for generation/evaluation runs."""
    return {
        "bm_sampler_type": config.bm_sampler_type,
        "bm_sampler_kwargs": config.bm_sampler_kwargs,
    }


def build_generator_for_eval(
    config: EvalConfig,
    *,
    device: torch.device,
    num_candidates: int,
    proposal_temperature: float = 0.0,
    proposal_noise_scale: float = 1.0,
    energy_temperature: float = 1.0,
    disable_resample: bool = False,
    resample_ratio: float = 0.25,
    resample_top_p: float = 0.95,
):
    """Builds one generator for baseline or guided ESM2 evaluation."""
    return (
        build_qdiffusion_protein(
            proposal_ckpt=config.proposal_ckpt,
            energy_ckpt=config.energy_ckpt,
            num_candidates=num_candidates,
            proposal_temperature=proposal_temperature,
            proposal_noise_scale=proposal_noise_scale,
            energy_temperature=energy_temperature,
            disable_resample=disable_resample,
            resample_ratio=resample_ratio,
            resample_top_p=resample_top_p,
            freeze_proposal=config.freeze_proposal,
            **build_bm_kwargs(config),
        )
        .eval()
        .to(device)
    )


def build_sa_eval_config(
    *,
    reference_fasta: Path,
    proposal_ckpt: str,
    energy_ckpt: str,
    guided_checkpoint: str,
    output_dir: Path,
    device: str,
    esm2_model: str = "esm2_t33_650M_UR50D",
    pair_mode: str = "order",
    pooling: str = "mean",
    batch_size: int = 1,
    max_records: int | None = 20,
    generation_steps: int = 500,
    seed: int = 42,
    freeze_proposal: bool = True,
    guided_num_candidates: int = 4,
    guided_proposal_temperature: float = 0.3,
    guided_proposal_noise_scale: float = 1.0,
    guided_energy_temperature: float = 1.25,
    guided_disable_resample: bool = False,
    guided_resample_ratio: float = 0.20,
    guided_resample_top_p: float = 0.90,
) -> EvalConfig:
    """Builds one ready-to-run SA-backed ESM2 evaluation config.

    Args:
        reference_fasta: Reference FASTA used for generation and evaluation.
        proposal_ckpt: Proposal-model checkpoint or model id.
        energy_ckpt: Energy-model checkpoint or model id.
        guided_checkpoint: Trained compact energy checkpoint.
        output_dir: Root output directory for generated/evaluation artifacts.
        device: Runtime device string.
        esm2_model: ESM2 model name from ``esm.pretrained``.
        pair_mode: Pairing strategy for reference/candidate sequences.
        pooling: Sequence embedding pooling strategy.
        batch_size: ESM2 embedding batch size.
        max_records: Optional reference-record limit.
        generation_steps: Number of generation steps per sequence.
        seed: Base random seed for generation.
        freeze_proposal: Whether the proposal backbone stays frozen.
        guided_num_candidates: Guided reranker candidate count.
        guided_proposal_temperature: Guided proposal temperature.
        guided_proposal_noise_scale: Guided proposal noise scale.
        guided_energy_temperature: Guided energy temperature.
        guided_disable_resample: Whether guided resampling is disabled.
        guided_resample_ratio: Guided repetition-resample threshold.
        guided_resample_top_p: Guided resample top-p cutoff.

    Returns:
        EvalConfig: An evaluation config that uses simulated annealing for BM
        hidden-state solving.
    """
    return EvalConfig(
        reference_fasta=reference_fasta,
        proposal_ckpt=proposal_ckpt,
        energy_ckpt=energy_ckpt,
        guided_checkpoint=guided_checkpoint,
        output_dir=output_dir,
        device=device,
        esm2_model=esm2_model,
        pair_mode=pair_mode,
        pooling=pooling,
        batch_size=batch_size,
        max_records=max_records,
        generation_steps=generation_steps,
        seed=seed,
        freeze_proposal=freeze_proposal,
        guided_num_candidates=guided_num_candidates,
        guided_proposal_temperature=guided_proposal_temperature,
        guided_proposal_noise_scale=guided_proposal_noise_scale,
        guided_energy_temperature=guided_energy_temperature,
        guided_disable_resample=guided_disable_resample,
        guided_resample_ratio=guided_resample_ratio,
        guided_resample_top_p=guided_resample_top_p,
        bm_sampler_type="sa",
        bm_sampler_kwargs=None,
    )


def build_cim_eval_config(
    *,
    reference_fasta: Path,
    proposal_ckpt: str,
    energy_ckpt: str,
    guided_checkpoint: str,
    output_dir: Path,
    device: str,
    task_name: str,
    project_no: str | None = None,
    task_mode: str | None = None,
    sample_number: int | None = None,
    tmp_dir: str | None = None,
    wait: bool = False,
    interval: int = 1,
    use_precision_reducer: bool = False,
    precision: int = 8,
    truncated_precision: int = 10,
    target_bits: int = 550,
    only_feasible_solution: bool = False,
    esm2_model: str = "esm2_t33_650M_UR50D",
    pair_mode: str = "order",
    pooling: str = "mean",
    batch_size: int = 1,
    max_records: int | None = 20,
    generation_steps: int = 500,
    seed: int = 42,
    freeze_proposal: bool = True,
    guided_num_candidates: int = 4,
    guided_proposal_temperature: float = 0.3,
    guided_proposal_noise_scale: float = 1.0,
    guided_energy_temperature: float = 1.25,
    guided_disable_resample: bool = False,
    guided_resample_ratio: float = 0.20,
    guided_resample_top_p: float = 0.90,
) -> EvalConfig:
    """Builds one ready-to-run CIM-backed ESM2 evaluation config.

    Args:
        reference_fasta: Reference FASTA used for generation and evaluation.
        proposal_ckpt: Proposal-model checkpoint or model id.
        energy_ckpt: Energy-model checkpoint or model id.
        guided_checkpoint: Trained compact energy checkpoint.
        output_dir: Root output directory for generated/evaluation artifacts.
        device: Runtime device string.
        task_name: CIM task name forwarded to ``CIMOptimizer``.
        project_no: Optional CIM project id.
        task_mode: Optional CIM task mode string such as ``"OPTIMIZATION"``
            or ``"SAMPLING"``.
        sample_number: Optional CIM sample count used for sampling mode.
        tmp_dir: Optional checkpoint/cache directory required by some CIM runs.
        wait: Whether to block until CIM execution finishes.
        interval: Polling interval in minutes.
        use_precision_reducer: Whether to wrap the CIM optimizer with
            ``PrecisionReducer``.
        precision: PrecisionReducer precision.
        truncated_precision: PrecisionReducer truncated precision.
        target_bits: PrecisionReducer target bit count.
        only_feasible_solution: PrecisionReducer feasibility flag.
        esm2_model: ESM2 model name from ``esm.pretrained``.
        pair_mode: Pairing strategy for reference/candidate sequences.
        pooling: Sequence embedding pooling strategy.
        batch_size: ESM2 embedding batch size.
        max_records: Optional reference-record limit.
        generation_steps: Number of generation steps per sequence.
        seed: Base random seed for generation.
        freeze_proposal: Whether the proposal backbone stays frozen.
        guided_num_candidates: Guided reranker candidate count.
        guided_proposal_temperature: Guided proposal temperature.
        guided_proposal_noise_scale: Guided proposal noise scale.
        guided_energy_temperature: Guided energy temperature.
        guided_disable_resample: Whether guided resampling is disabled.
        guided_resample_ratio: Guided repetition-resample threshold.
        guided_resample_top_p: Guided resample top-p cutoff.

    Returns:
        EvalConfig: An evaluation config that routes BM hidden-state solving
        through ``CIMOptimizer``.
    """
    return EvalConfig(
        reference_fasta=reference_fasta,
        proposal_ckpt=proposal_ckpt,
        energy_ckpt=energy_ckpt,
        guided_checkpoint=guided_checkpoint,
        output_dir=output_dir,
        device=device,
        esm2_model=esm2_model,
        pair_mode=pair_mode,
        pooling=pooling,
        batch_size=batch_size,
        max_records=max_records,
        generation_steps=generation_steps,
        seed=seed,
        freeze_proposal=freeze_proposal,
        guided_num_candidates=guided_num_candidates,
        guided_proposal_temperature=guided_proposal_temperature,
        guided_proposal_noise_scale=guided_proposal_noise_scale,
        guided_energy_temperature=guided_energy_temperature,
        guided_disable_resample=guided_disable_resample,
        guided_resample_ratio=guided_resample_ratio,
        guided_resample_top_p=guided_resample_top_p,
        bm_sampler_type="cim",
        bm_sampler_kwargs={
            "task_name": task_name,
            "wait": wait,
            "interval": interval,
            "project_no": project_no,
            "task_mode": task_mode,
            "sample_number": sample_number,
            "use_precision_reducer": use_precision_reducer,
            "precision": precision,
            "truncated_precision": truncated_precision,
            "target_bits": target_bits,
            "only_feasible_solution": only_feasible_solution,
            "tmp_dir": tmp_dir,
        },
    )


def maybe_limit_records(
    records: list[tuple[str, str]], max_records: int | None
) -> list[tuple[str, str]]:
    """Optionally truncates the reference set to keep one run manageable.

    Args:
        records: Input FASTA records.
        max_records: Optional limit on record count.

    Returns:
        list[tuple[str, str]]: Either the full record list or its truncated prefix.
    """
    if max_records is None:
        return records
    return records[:max_records]


# ---------------------------------------------------------------------------
# Generation helpers
# This section mirrors the old generate_dplm_ebm.py behavior:
# build an all-mask input of the requested length, run baseline/guided
# generators, and write the generated FASTA files.
# ---------------------------------------------------------------------------
def build_full_mask_input(generator, sequence_length: int) -> torch.Tensor:
    """Builds one all-mask input tensor with the requested residue length.

    Args:
        generator: Configured ``QDiffusion`` instance.
        sequence_length: Residue length excluding special tokens.

    Returns:
        torch.Tensor: Input token tensor filled with mask tokens at residue positions.
    """
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
    """Generates one sequence per reference record and writes one FASTA file.

    Args:
        generator: Configured ``QDiffusion`` instance.
        records: Reference FASTA records.
        max_steps: Decode step count for each generation.
        seed_base: Base random seed; each record offsets from this value.
        output_fasta_path: FASTA output path.
        label: Logging label for the generation run.

    Returns:
        list[tuple[str, str]]: Generated FASTA records aligned to the input order.
    """
    generated_records: list[tuple[str, str]] = []
    print(
        f"[{label}] starting generation for {len(records)} sequences "
        f"with max_steps={max_steps}"
    )

    for index, (header, sequence) in enumerate(records, start=1):
        sequence = normalize_sequence(sequence)
        # Follow generate_dplm_ebm.py: generation starts from an all-mask
        # sequence with the same residue length as the reference.
        input_tokens = build_full_mask_input(generator, len(sequence))
        # Only special tokens are fixed at this stage; residue positions remain
        # editable because they are mask_id.
        partial_masks = input_tokens.ne(generator.mask_id)
        seed = seed_base + index
        seed_torch(seed)

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


def generate_candidate_fastas(
    *,
    config: EvalConfig,
    reference_records: list[tuple[str, str]],
    device: torch.device,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generates baseline and guided FASTA files before ESM2 evaluation.

    Args:
        config: Top-level evaluation config.
        reference_records: Reference FASTA records used to drive generation.
        device: Runtime device for generation models.
        output_dir: Root output directory for generated artifacts.

    Returns:
        tuple[Path, Path]: A tuple ``(baseline_path, guided_path)`` for the generated FASTA files.

    Raises:
        SystemExit: If required checkpoint config is missing.
    """
    if (
        not config.proposal_ckpt
        or not config.energy_ckpt
        or not config.guided_checkpoint
    ):
        raise SystemExit(
            "proposal_ckpt, energy_ckpt, and guided_checkpoint must be set in main()."
        )

    generation_dir = output_dir / "generated"
    generation_dir.mkdir(parents=True, exist_ok=True)
    generation_cfg = GenerationConfig(
        proposal_ckpt=config.proposal_ckpt,
        energy_ckpt=config.energy_ckpt,
        guided_checkpoint=config.guided_checkpoint,
        generation_steps=config.generation_steps,
        seed=config.seed,
        freeze_proposal=config.freeze_proposal,
        guided_num_candidates=config.guided_num_candidates,
        guided_proposal_temperature=config.guided_proposal_temperature,
        guided_proposal_noise_scale=config.guided_proposal_noise_scale,
        guided_energy_temperature=config.guided_energy_temperature,
        guided_disable_resample=config.guided_disable_resample,
        guided_resample_ratio=config.guided_resample_ratio,
        guided_resample_top_p=config.guided_resample_top_p,
        bm_sampler_type=config.bm_sampler_type,
        bm_sampler_kwargs=config.bm_sampler_kwargs,
    )
    save_json(generation_dir / "generation_config.json", asdict(generation_cfg))
    print(f"Saved generation config to: {generation_dir / 'generation_config.json'}")

    baseline_path = generation_dir / "baseline_generated_sequences.fasta"
    guided_path = generation_dir / "guided_generated_sequences.fasta"

    # Baseline mode: proposal-only behavior, aligned with the old script family.
    print("Building baseline generator...")
    baseline_generator = build_generator_for_eval(
        config,
        device=device,
        num_candidates=1,
    )
    run_generation_over_records(
        baseline_generator,
        reference_records,
        max_steps=config.generation_steps,
        seed_base=config.seed,
        output_fasta_path=baseline_path,
        label="baseline",
    )

    # Guided mode: same proposal backbone family, but energy-side weights are
    # restored from the trained compact checkpoint.
    print("Building guided generator...")
    guided_generator = build_generator_for_eval(
        config,
        device=device,
        num_candidates=config.guided_num_candidates,
        proposal_temperature=config.guided_proposal_temperature,
        proposal_noise_scale=config.guided_proposal_noise_scale,
        energy_temperature=config.guided_energy_temperature,
        disable_resample=config.guided_disable_resample,
        resample_ratio=config.guided_resample_ratio,
        resample_top_p=config.guided_resample_top_p,
    )
    load_trained_energy_weights(guided_generator, config.guided_checkpoint, device)
    run_generation_over_records(
        guided_generator,
        reference_records,
        max_steps=config.generation_steps,
        seed_base=config.seed,
        output_fasta_path=guided_path,
        label="guided",
    )

    return baseline_path, guided_path


# ---------------------------------------------------------------------------
# Pairing helpers
# This section decides how reference/generated records are aligned before
# embedding comparison.
# ---------------------------------------------------------------------------
def pair_records_by_order(
    reference_records: list[tuple[str, str]],
    candidate_records: list[tuple[str, str]],
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Pairs records positionally, truncating to the shorter side."""
    paired_count = min(len(reference_records), len(candidate_records))
    return list(zip(reference_records[:paired_count], candidate_records[:paired_count]))


def pair_records_by_header(
    reference_records: list[tuple[str, str]],
    candidate_records: list[tuple[str, str]],
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Pairs records by exact FASTA header match, preserving reference order."""
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
    """Yields fixed-size slices from one list."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


# ---------------------------------------------------------------------------
# ESM2 embedding helpers
# This section loads ESM2 and converts sequences into pooled sequence-level
# embeddings used for distance comparison.
# ---------------------------------------------------------------------------
def load_esm2_model(
    model_name: str, device: torch.device
) -> tuple[torch.nn.Module, object]:
    """Loads one supported ESM2 pretrained model from the esm package."""
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
    """Embeds each sequence into one pooled ESM2 representation."""
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

                # Slice away BOS/EOS and pool only residue-token representations
                # unless the caller explicitly asks for BOS/CLS/EOS pooling.
                residue_repr = token_representations[row_index, 1 : seq_len + 1]
                if pooling == "mean":
                    pooled = residue_repr.mean(dim=0)
                elif pooling == "cls":
                    pooled = token_representations[row_index, 0]
                elif pooling == "bos":
                    pooled = token_representations[row_index, 0]
                elif pooling == "eos":
                    eos_index = int(lengths[row_index].item()) - 1
                    pooled = token_representations[row_index, eos_index]
                else:
                    raise ValueError(f"Unsupported pooling mode: {pooling}")

                embeddings[header] = pooled.detach().cpu()

    return embeddings


# ---------------------------------------------------------------------------
# Distance computation helpers
# This section turns ESM2 embeddings into per-pair distances and aggregate
# summary metrics.
# ---------------------------------------------------------------------------
def summarize_distances(label: str, rows: list[PairDistanceRow]) -> DistanceSummary:
    """Builds one aggregate summary from per-pair rows."""
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
    """Evaluates one candidate FASTA against one reference FASTA."""
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
        # Cosine distance is 1 - cosine similarity; lower means embedding-wise
        # closer to the reference sequence.
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


# ---------------------------------------------------------------------------
# Output helpers
# This section writes CSV/JSON/Markdown artifacts for later inspection.
# ---------------------------------------------------------------------------
def write_rows_csv(path: Path, rows: list[PairDistanceRow]) -> None:
    """Writes per-pair distance rows to CSV."""
    write_csv_rows(path, [asdict(row) for row in rows])


def write_summary_json(path: Path, summary: DistanceSummary) -> None:
    """Writes one summary object to JSON."""
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
    """Writes one human-readable markdown report."""
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


# ---------------------------------------------------------------------------
# Main entrypoint
# High-level flow:
# 1. read reference FASTA
# 2. generate baseline/guided FASTA from all-mask inputs
# 3. embed reference and generated sequences with ESM2
# 4. compute distances and write reports
# ---------------------------------------------------------------------------
def main() -> None:
    """Runs one local/server generation+evaluation pass with in-file config."""
    config = build_sa_eval_config(
        reference_fasta=default_fasta_path(),
        proposal_ckpt="/data2/wwx/models/dplm_150m",
        energy_ckpt="/data2/wwx/models/dplm_150m",
        guided_checkpoint="ckpt/best_epoch_9.pt",
        output_dir=default_outputs_root() / "esm2_distance_eval",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        esm2_model="esm2_t33_650M_UR50D",
        pair_mode="order",
        pooling="mean",
        batch_size=1,
        max_records=20,
        generation_steps=500,
        seed=42,
        freeze_proposal=True,
        guided_num_candidates=4,
        guided_proposal_temperature=0.3,
        guided_proposal_noise_scale=1.0,
        guided_energy_temperature=1.25,
        guided_disable_resample=False,
        guided_resample_ratio=0.20,
        guided_resample_top_p=0.90,
    )
    # Switch to the CIM template when you want generation/evaluation to use the
    # same remote CIM sampling path as training.
    #
    # config = build_cim_eval_config(
    #     reference_fasta=default_fasta_path(),
    #     proposal_ckpt="/data2/wwx/models/dplm_150m",
    #     energy_ckpt="/data2/wwx/models/dplm_150m",
    #     guided_checkpoint="ckpt/best_epoch_9.pt",
    #     output_dir=default_outputs_root() / "esm2_distance_eval",
    #     device="cuda:0" if torch.cuda.is_available() else "cpu",
    #     task_name="qdiffusion_bm",
    #     project_no="YOUR_PROJECT_ID",
    #     task_mode="OPTIMIZATION",
    #     tmp_dir="./tmp",
    #     wait=False,
    # )

    device = torch.device(config.device)
    output_dir: Path = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_records = [
        (header, normalize_sequence(sequence))
        for header, sequence in read_fasta_records(config.reference_fasta)
    ]
    reference_records = maybe_limit_records(reference_records, config.max_records)
    print(f"Loaded {len(reference_records)} reference sequences from: {config.reference_fasta}")
    baseline_path, guided_path = generate_candidate_fastas(
        config=config,
        reference_records=reference_records,
        device=device,
        output_dir=output_dir,
    )
    baseline_records = [
        (header, normalize_sequence(sequence))
        for header, sequence in read_fasta_records(baseline_path)
    ]
    guided_records = [
        (header, normalize_sequence(sequence))
        for header, sequence in read_fasta_records(guided_path)
    ]

    print("Starting ESM2 evaluation stage...")
    model, alphabet = load_esm2_model(config.esm2_model, device)

    reference_embeddings = embed_sequences(
        reference_records,
        model=model,
        alphabet=alphabet,
        device=device,
        batch_size=config.batch_size,
        pooling=config.pooling,
    )

    summaries: list[DistanceSummary] = []
    baseline_embeddings = embed_sequences(
        baseline_records,
        model=model,
        alphabet=alphabet,
        device=device,
        batch_size=config.batch_size,
        pooling=config.pooling,
    )
    baseline_rows, baseline_summary = evaluate_candidate_set(
        label="baseline",
        reference_records=reference_records,
        candidate_records=baseline_records,
        reference_embeddings=reference_embeddings,
        candidate_embeddings=baseline_embeddings,
        pair_mode=config.pair_mode,
    )
    write_rows_csv(output_dir / "baseline_pair_distances.csv", baseline_rows)
    write_summary_json(output_dir / "baseline_summary.json", baseline_summary)
    summaries.append(baseline_summary)

    guided_embeddings = embed_sequences(
        guided_records,
        model=model,
        alphabet=alphabet,
        device=device,
        batch_size=config.batch_size,
        pooling=config.pooling,
    )
    guided_rows, guided_summary = evaluate_candidate_set(
        label="guided",
        reference_records=reference_records,
        candidate_records=guided_records,
        reference_embeddings=reference_embeddings,
        candidate_embeddings=guided_embeddings,
        pair_mode=config.pair_mode,
    )
    write_rows_csv(output_dir / "guided_pair_distances.csv", guided_rows)
    write_summary_json(output_dir / "guided_summary.json", guided_summary)
    summaries.append(guided_summary)

    write_report(
        output_dir / "REPORT.md",
        reference_path=config.reference_fasta,
        baseline_path=baseline_path,
        guided_path=guided_path,
        summaries=summaries,
        model_name=config.esm2_model,
        pair_mode=config.pair_mode,
        pooling=config.pooling,
    )
    print(f"Saved evaluation report to: {output_dir / 'REPORT.md'}")

    print(f"Saved ESM2 distance evaluation to: {output_dir}")
    for summary in summaries:
        print(
            f"[{summary.label}] pairs={summary.paired_count} "
            f"mean_cosine_distance={summary.mean_cosine_distance:.6f} "
            f"mean_l2_distance={summary.mean_l2_distance:.6f}"
        )


if __name__ == "__main__":
    main()
