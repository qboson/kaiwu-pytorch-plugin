"""Shared helpers for the Nemotron example CLIs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL dataset of ``problem``/``answer`` rows.

    Args:
        path: Path to the JSONL dataset file.

    Returns:
        Parsed rows in file order; blank lines are skipped.

    Raises:
        ValueError: If a line is not a JSON object with ``problem`` and
            ``answer`` keys, or if the dataset is empty.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "problem" not in row or "answer" not in row:
                raise ValueError(f"line {line_number} must contain problem and answer")
            rows.append(row)
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def prompt_ids(tokenizer: Any, problem: str, device: str) -> torch.Tensor:
    """Tokenize one math problem with the shared instruction template.

    Args:
        tokenizer: Tokenizer providing ``apply_chat_template``.
        problem: Raw problem text appended to the shared instruction.
        device: Device string the returned tensor is moved to.

    Returns:
        Prompt token ids shaped ``[1, seq_len]`` placed on ``device``.
    """
    instruction = (
        "Solve the following math problem. Put the final answer inside "
        "\\boxed{} at the very end.\n\n"
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction + problem}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def load_nemotron(path: str, device: str) -> tuple[Any, Any]:
    """Load the frozen remote-code Nemotron tokenizer and model.

    Args:
        path: Model checkpoint directory loaded with remote code trusted.
        device: Device string the model is moved to.

    Returns:
        Tuple ``(tokenizer, model)`` with the model in eval mode and
        ``bfloat16`` dtype.
    """
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        path, trust_remote_code=True, dtype=torch.bfloat16
    )
    return tokenizer, model.to(device).eval()


def file_identity(path: Path) -> dict[str, Any]:
    """Return the path, size, and SHA-256 digest used by resume checks.

    Args:
        path: File to hash and stat.

    Returns:
        Dict with the resolved ``path``, byte ``size``, and ``sha256``
        hex digest of the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON payload.

    Args:
        path: Destination path; parent directories are created.
        payload: JSON-serializable object written with ``indent=2``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    """Atomically write a ``torch.save`` payload.

    Args:
        path: Destination path; parent directories are created.
        payload: Object passed to ``torch.save``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def native_generate(
    session: Any,
    inputs: torch.Tensor,
    *,
    max_new_tokens: int,
    block_length: int,
    threshold: float,
    eos_token_id: int,
) -> tuple[torch.Tensor, int]:
    """Run one native generation call with the shared sampling contract.

    The example always decodes greedily; ``temperature=0.0`` is fixed here so
    that every CLI issues identical native generate calls.

    Args:
        session: Native generation session exposing ``generate``.
        inputs: Prompt token ids shaped ``[batch, seq_len]``.
        max_new_tokens: Maximum number of newly generated tokens.
        block_length: Block size used by the native block-decoding loop.
        threshold: Confidence threshold of the native decoder.
        eos_token_id: Token id that stops generation.

    Returns:
        Tuple ``(sequences, nfe)`` with generated token ids and the number
        of function evaluations reported by the native session.
    """
    return session.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        threshold=threshold,
        temperature=0.0,
        eos_token_id=eos_token_id,
    )
