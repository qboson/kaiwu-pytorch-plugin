"""Shared I/O helpers for Q-Diffusion DPLM example workflows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CASE_ROOT = Path(__file__).resolve().parents[1]


def default_fasta_path() -> Path:
    """Returns the bundled FASTA path used by the example workflows."""
    return CASE_ROOT / "data" / "UP000005640_9606.fasta"


def default_outputs_root() -> Path:
    """Returns the root output directory for DPLM example artifacts."""
    return CASE_ROOT / "outputs"


def save_json(path: Path, payload: Any) -> None:
    """Writes stable JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_markdown(path: Path, lines: list[str]) -> None:
    """Writes Markdown content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def read_fasta_records(fasta_path: Path) -> list[tuple[str, str]]:
    """Reads all FASTA records from one file."""
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
    """Writes FASTA records to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, (header, sequence) in enumerate(records, start=1):
            handle.write(f">seq_{index} {header}\n")
            handle.write(sequence + "\n")


def write_tsv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes row dictionaries to one TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes row dictionaries to one CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_sequence(sequence: str) -> str:
    """Converts decoded sequence text into compact FASTA-friendly text."""
    return sequence.replace(" ", "").strip()


def normalize_decoded_sequence(sequence: str) -> str:
    """Alias kept for training/rerun scripts that already use this wording."""
    return normalize_sequence(sequence)
