"""Evaluate ``QDiffusion`` experiment outputs with ESM2 embedding distances.

This script follows one fixed workflow:

1. read the reference FASTA
2. generate baseline sequences from all-mask inputs
3. generate guided sequences from all-mask inputs
4. embed reference/baseline/guided sequences with ESM2
5. compare baseline/guided against the reference set

Edit the config block inside ``main()`` before running, from the repository
root after ``pip install -e .``:

    python -m example.qdiffusion.dplm.workflows.esm2_eval
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..utils.dplm_builder import build_qdiffusion
from ..utils.io import (
    default_fasta_path,
    default_outputs_root,
    normalize_sequence,
    read_fasta_records,
    save_json,
    write_csv_rows,
)
from ..utils.runtime import load_trained_energy_weights
from .esm2_eval_helpers import (
    DistanceSummary,
    embed_sequences,
    evaluate_candidate_set,
    load_esm2_model,
    run_masked_generation_over_records,
    write_report,
)


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
        build_qdiffusion(
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
        ValueError: If required checkpoint config is missing.
    """
    if (
        not config.proposal_ckpt
        or not config.energy_ckpt
        or not config.guided_checkpoint
    ):
        raise ValueError(
            "proposal_ckpt, energy_ckpt, and guided_checkpoint must be set in main()."
        )

    generation_dir = output_dir / "generated"
    generation_dir.mkdir(parents=True, exist_ok=True)
    generation_cfg = {
        "proposal_ckpt": config.proposal_ckpt,
        "energy_ckpt": config.energy_ckpt,
        "guided_checkpoint": config.guided_checkpoint,
        "generation_steps": config.generation_steps,
        "seed": config.seed,
        "freeze_proposal": config.freeze_proposal,
        "guided": {
            "num_candidates": config.guided_num_candidates,
            "proposal_temperature": config.guided_proposal_temperature,
            "proposal_noise_scale": config.guided_proposal_noise_scale,
            "energy_temperature": config.guided_energy_temperature,
            "disable_resample": config.guided_disable_resample,
            "resample_ratio": config.guided_resample_ratio,
            "resample_top_p": config.guided_resample_top_p,
        },
        "bm_sampler_type": config.bm_sampler_type,
        "bm_sampler_kwargs": config.bm_sampler_kwargs,
    }
    save_json(generation_dir / "generation_config.json", generation_cfg)
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
    run_masked_generation_over_records(
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
    run_masked_generation_over_records(
        guided_generator,
        reference_records,
        max_steps=config.generation_steps,
        seed_base=config.seed,
        output_fasta_path=guided_path,
        label="guided",
    )

    return baseline_path, guided_path


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
    # Edit the values below before running. The guided checkpoint is the
    # energy-side artifact produced by ``workflows/train.py`` (best_epoch_*.pt).
    config = build_sa_eval_config(
        reference_fasta=default_fasta_path(),
        proposal_ckpt="airkingbd/dplm_150m",
        energy_ckpt="airkingbd/dplm_150m",
        guided_checkpoint="checkpoints/best.pt",
        output_dir=default_outputs_root() / "esm2_distance_eval",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    device = torch.device(config.device)
    output_dir: Path = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_records = [
        (header, normalize_sequence(sequence))
        for header, sequence in read_fasta_records(config.reference_fasta)
    ]
    if config.max_records is not None:
        reference_records = reference_records[: config.max_records]
    print(
        f"Loaded {len(reference_records)} reference sequences from: {config.reference_fasta}"
    )
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
    )
    write_csv_rows(
        output_dir / "baseline_pair_distances.csv",
        [asdict(row) for row in baseline_rows],
    )
    save_json(output_dir / "baseline_summary.json", asdict(baseline_summary))
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
    )
    write_csv_rows(
        output_dir / "guided_pair_distances.csv",
        [asdict(row) for row in guided_rows],
    )
    save_json(output_dir / "guided_summary.json", asdict(guided_summary))
    summaries.append(guided_summary)

    write_report(
        output_dir / "REPORT.md",
        reference_path=config.reference_fasta,
        baseline_path=baseline_path,
        guided_path=guided_path,
        summaries=summaries,
        model_name=config.esm2_model,
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
