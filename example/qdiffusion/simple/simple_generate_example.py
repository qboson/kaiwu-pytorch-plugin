# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Minimal generation example for ``QDiffusion``.

This script keeps only the smallest useful inference path:

1. build generator
2. tokenize one protein sequence
3. call ``generate()``
4. decode the generated tokens
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _example_bootstrap import ensure_repo_src_on_path
import torch

ensure_repo_src_on_path()

from dplm.utils.dplm_builder import build_qdiffusion

# Path and sequence helpers.

EXAMPLE_DIR = Path(__file__).resolve().parent
CASE_ROOT = EXAMPLE_DIR.parent


def default_fasta_path() -> Path:
    """Returns the bundled example FASTA path.

    Returns:
        Path: Path to the bundled FASTA file used by this example.
    """
    return CASE_ROOT / "data" / "UP000005640_9606.fasta"


def first_usable_sequence(
    fasta_path: Path,
    *,
    min_length: int,
    max_length: int,
) -> tuple[str, str]:
    """Returns the first sequence within the requested length range.

    Args:
        fasta_path: FASTA file to scan.
        min_length: Minimum acceptable sequence length.
        max_length: Maximum acceptable sequence length.

    Returns:
        tuple[str, str]: A ``(header, sequence)`` pair.

    Raises:
        ValueError: If no usable sequence exists in the FASTA file.
    """
    header = ""
    sequence_parts: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    sequence = "".join(sequence_parts)
                    if min_length <= len(sequence) <= max_length:
                        return header, sequence
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)

    if header:
        sequence = "".join(sequence_parts)
        if min_length <= len(sequence) <= max_length:
            return header, sequence

    raise ValueError("No usable sequence found in FASTA.")


def decode_tokens(generator, tokens: torch.Tensor) -> str:
    """Decodes one generated token tensor into sequence text.

    Args:
        generator: Configured ``QDiffusion`` instance.
        tokens: Generated token tensor.

    Returns:
        str: Decoded protein sequence text without spaces.
    """
    decoded = generator.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return decoded.replace(" ", "").strip()


# Example entrypoint.


def parse_args(argv=None):
    """Parses command-line arguments for this example.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Minimal QDiffusion example."
    )
    parser.add_argument(
        "--fasta-path",
        default=None,
        help=(
            "FASTA file to read sequences from. Defaults to the bundled "
            "example proteome location, which is not shipped in the "
            "repository and must be downloaded first."
        ),
    )
    return parser.parse_args(argv)


def resolve_fasta_path(cli_path):
    """Resolves the FASTA input and fails fast when it is unavailable.

    Args:
        cli_path: Path passed via ``--fasta-path``, or ``None`` for the
            documented default location.

    Returns:
        Path: The FASTA file the example should read.

    Raises:
        FileNotFoundError: If the resolved FASTA file does not exist. Raised
            before any pretrained model is built so the user gets an
            actionable message instead of a failure after a long download.
    """
    fasta_path = Path(cli_path) if cli_path is not None else default_fasta_path()
    if not fasta_path.is_file():
        raise FileNotFoundError(
            f"FASTA file not found: {fasta_path}. The example proteome is "
            "not bundled with the repository; download it (see "
            "example/qdiffusion/README.md) or pass --fasta-path to point at "
            "a local FASTA file."
        )
    return fasta_path


def main(argv=None) -> None:
    """Runs a tiny example generation loop.

    The example loads one pretrained ``QDiffusion`` model, reads the first
    usable sequence from the bundled FASTA, and prints the generated result.
    """
    args = parse_args(argv)
    fasta_path = resolve_fasta_path(args.fasta_path)
    proposal_ckpt = "airkingbd/dplm_150m"
    energy_ckpt = "airkingbd/dplm_150m"
    max_steps = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = build_qdiffusion(
        proposal_ckpt=proposal_ckpt,
        energy_ckpt=energy_ckpt,
        num_candidates=4,
        proposal_temperature=0.3,
        energy_temperature=1.25,
        resample_ratio=0.20,
        resample_top_p=0.90,
        freeze_proposal=True,
    ).to(device)
    generator.eval()

    header, sequence = first_usable_sequence(
        fasta_path,
        min_length=50,
        max_length=256,
    )
    encoded = generator.tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
    )
    # In the public API the caller provides a clean token sequence and
    # ``generate()`` handles the iterative noisy-state decoding internally.
    input_tokens = encoded["input_ids"].to(generator.device)

    with torch.no_grad():
        generated_tokens = generator.generate(
            input_tokens=input_tokens,
            max_steps=max_steps,
        )

    generated_sequence = decode_tokens(generator, generated_tokens)
    print(f"header={header}")
    print(f"reference_length={len(sequence)}")
    print(f"generated_length={len(generated_sequence)}")
    print(f"generated_sequence={generated_sequence}")


if __name__ == "__main__":
    main()
