# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Minimal training example for ``QDiffusion``.

This script intentionally keeps only the smallest useful training path:

1. build generator
2. read a few FASTA sequences
3. tokenize one mini-batch
4. call ``objective()``
5. optimize ``energy_objective.mean()``
"""

from __future__ import annotations

from pathlib import Path

from _example_bootstrap import ensure_repo_src_on_path
import torch
from torch.optim import AdamW

ensure_repo_src_on_path()

from dplm.dplm_builder import build_dplm_qdiffusion

# Path and data helpers.

EXAMPLE_DIR = Path(__file__).resolve().parent
CASE_ROOT = EXAMPLE_DIR.parent


def default_fasta_path() -> Path:
    """Returns the bundled example FASTA path.

    Returns:
        Path: Path to the bundled FASTA file used by this example.
    """
    return CASE_ROOT / "data" / "UP000005640_9606.fasta"


def read_fasta_records(fasta_path: Path) -> list[tuple[str, str]]:
    """Reads FASTA records from one file.

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


def select_records(
    records: list[tuple[str, str]],
    *,
    min_length: int,
    max_length: int,
    max_records: int,
) -> list[tuple[str, str]]:
    """Selects the first few usable records by length.

    Args:
        records: FASTA records to filter.
        min_length: Minimum acceptable sequence length.
        max_length: Maximum acceptable sequence length.
        max_records: Maximum number of records to keep.

    Returns:
        list[tuple[str, str]]: The filtered prefix of usable FASTA records.
    """
    selected: list[tuple[str, str]] = []
    for header, sequence in records:
        if len(sequence) < min_length or len(sequence) > max_length:
            continue
        selected.append((header, sequence))
        if len(selected) >= max_records:
            break
    return selected


# Example entrypoint.

def main() -> None:
    """Runs a tiny example training loop.

    The example loads one pretrained ``QDiffusion`` model, samples a small
    mini-batch from the bundled FASTA, and optimizes the EBM objective for a
    few steps.
    """
    proposal_ckpt = "airkingbd/dplm_150m"
    energy_ckpt = "airkingbd/dplm_150m"
    fasta_path = default_fasta_path()
    batch_size = 2
    num_steps = 3
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = build_dplm_qdiffusion(
        proposal_ckpt=proposal_ckpt,
        energy_ckpt=energy_ckpt,
        num_candidates=4,
        freeze_proposal=True,
    ).to(device)

    records = select_records(
        read_fasta_records(fasta_path),
        min_length=50,
        max_length=256,
        max_records=batch_size,
    )
    sequences = [sequence for _, sequence in records]

    encoded = generator.tokenizer(
        sequences,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )
    targets = encoded["input_ids"].to(generator.device)

    optimizer = AdamW(
        [parameter for parameter in generator.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )

    generator.train()
    if getattr(generator, "proposal_model", None) is not None:
        generator.proposal_model.eval()

    for step in range(1, num_steps + 1):
        outputs = generator.objective({"targets": targets})
        loss = outputs["energy_objective"].mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(
            f"step={step} "
            f"loss={loss.item():.6f} "
            f"logits_shape={tuple(outputs['logits'].shape)} "
            f"masked_positions={int(outputs['loss_mask'].sum().item())}"
        )


if __name__ == "__main__":
    main()
