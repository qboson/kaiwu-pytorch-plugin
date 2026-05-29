# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint rerun workflow for guided ``QDiffusion`` experiments."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from _example_bootstrap import ensure_repo_src_on_path
import torch

ensure_repo_src_on_path()

from dplm_builder import build_dplm_qdiffusion
try:
    from .utils.io import (
        default_outputs_root,
        normalize_decoded_sequence,
        read_fasta_records,
        save_json,
        save_markdown,
        write_fasta_records,
        write_tsv_rows,
    )
    from .utils.metrics import (
        QualitySummary,
        compare_generation_sets,
        evaluate_generation_quality,
        save_quality_summary,
        sequence_identity,
    )
    from .utils.runtime import encode_sequence, load_trained_energy_weights
except ImportError:  # pragma: no cover - direct script-path compatibility
    from utils.io import (
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
        sequence_identity,
    )
    from utils.runtime import encode_sequence, load_trained_energy_weights

os.environ.setdefault("BYPROT_EAGER_IMPORTS", "0")


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
    write_tsv_rows(output_dir / f"{label}_generation_summary.tsv", rows)
    return generated_records, rows


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
    save_markdown(path, lines)


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
