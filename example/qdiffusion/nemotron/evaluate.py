"""NeMo-compatible MATH scoring and matched Native/BM evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from .common.answers import last_boxed_content, math_equivalent, require_math_verify
from .common.runtime import (
    atomic_json,
    file_identity,
    load_nemotron,
    native_generate,
    prompt_ids,
    read_jsonl,
)
from .generation.guidance import load_nemotron_guidance
from .generation.proposal import NativeGenerationSession


# Matched evaluation CLI with append-only resume.


PARTIAL_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    """Parses evaluation command-line arguments.

    Returns:
        argparse.Namespace: Parsed options; see ``--help`` and the README for
        the reference invocation of every flag.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-jsonl", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--strategies", nargs="+", choices=("native", "bm"), default=("native", "bm")
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--energy-lambda", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _file_identity(path: Path | None) -> dict[str, Any] | None:
    """Builds the dataset/checkpoint identity keyed by file name.

    Keeps the field shape already persisted in existing partial meta files so
    their resume fingerprints stay valid.

    Args:
        path: File to identify, or ``None``.

    Returns:
        Dict with ``name``, ``size``, and ``sha256`` keys, or ``None`` when
        ``path`` is ``None``.
    """
    if path is None:
        return None
    identity = file_identity(path)
    return {
        "name": Path(identity["path"]).name,
        "size": identity["size"],
        "sha256": identity["sha256"],
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    """Hashes a config payload into a stable resume fingerprint.

    Args:
        payload: JSON-serializable configuration mapping.

    Returns:
        SHA-256 hex digest of the canonical JSON encoding.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_partial_items(path: Path) -> list[dict[str, Any]]:
    """Load an append-only partial, repairing only a torn final write."""

    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    items: list[dict[str, Any]] = []
    valid_bytes = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            valid_bytes += len(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            is_torn_tail = line_number == len(lines) and not line.endswith(
                (b"\n", b"\r")
            )
            if not is_torn_tail:
                raise ValueError(
                    f"invalid partial JSON at line {line_number} in {path}"
                ) from None
            with path.open("r+b") as stream:
                stream.truncate(valid_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            break
        if record.get("schema_version") != PARTIAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported partial schema at line {line_number}")
        if not isinstance(record.get("item"), dict):
            raise ValueError(f"partial line {line_number} has no item object")
        items.append(record["item"])
        valid_bytes += len(line)
    return items


def _partial_store(
    args: argparse.Namespace, strategy: str
) -> tuple[Path, list[dict[str, Any]], str]:
    """Opens or initializes the append-only partial store for one strategy.

    Args:
        args: Parsed CLI options.

        strategy: Evaluation strategy, ``native`` or ``bm``.

    Returns:
        Tuple ``(items_path, items, fingerprint)`` with the JSONL path, the
        items completed so far, and the active configuration fingerprint.

    Raises:
        FileExistsError: If partial output exists and ``--resume`` is off.

        ValueError: If the stored fingerprint does not match the current
            configuration.
    """
    config = {
        "schema_version": PARTIAL_SCHEMA_VERSION,
        "strategy": strategy,
        "model": args.model,
        "dataset": _file_identity(args.dataset_jsonl),
        "checkpoint": _file_identity(args.checkpoint if strategy == "bm" else None),
        "max_new_tokens": args.max_new_tokens,
        "block_length": args.block_length,
        "threshold": args.threshold,
        "num_candidates": args.num_candidates if strategy == "bm" else 1,
        "energy_lambda": args.energy_lambda if strategy == "bm" else None,
        "seed": args.seed,
        "limit": args.limit,
    }
    fingerprint = _fingerprint(config)
    meta_path = args.output_dir / "partial" / f"{strategy}.meta.json"
    items_path = args.output_dir / "partial" / f"{strategy}.items.jsonl"
    if meta_path.exists() or items_path.exists():
        if not args.resume or not meta_path.exists() or not items_path.exists():
            raise FileExistsError(
                f"existing {strategy} partial output requires --resume"
            )
        metadata = json.loads(meta_path.read_text())
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError(f"partial configuration mismatch for {strategy}")
    else:
        atomic_json(meta_path, {"fingerprint": fingerprint, "config": config})
        items_path.parent.mkdir(parents=True, exist_ok=True)
        items_path.touch()
    items = _load_partial_items(items_path)
    return items_path, items, fingerprint


def _append_item(path: Path, item: dict[str, Any]) -> None:
    """Appends one item record to a partial JSONL file.

    Args:
        path: Destination JSONL file.

        item: Evaluation item to persist; the record is flushed and
            ``fsync``-ed before returning.
    """
    record = {"schema_version": PARTIAL_SCHEMA_VERSION, "item": item}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _summarize(
    strategy: str, items: list[dict[str, Any]], fingerprint: str
) -> dict[str, Any]:
    """Aggregates completed items into a result summary.

    Args:
        strategy: Evaluation strategy the items belong to.

        items: Completed evaluation items.

        fingerprint: Configuration fingerprint recorded with the summary.

    Returns:
        Dict with accuracy, token/NFE averages, elapsed time, and the full
        item list.
    """
    return {
        "strategy": strategy,
        "total": len(items),
        "correct": sum(bool(item["correct"]) for item in items),
        "accuracy": 100.0 * sum(bool(item["correct"]) for item in items) / len(items),
        "avg_tokens": sum(item["num_tokens"] for item in items) / len(items),
        "avg_nfe": sum(item["nfe"] for item in items) / len(items),
        "elapsed_seconds": sum(item["elapsed_seconds"] for item in items),
        "partial_fingerprint": fingerprint,
        "items": items,
    }


def main() -> None:
    """Runs matched Native/BM evaluation with append-only resume."""
    args = parse_args()
    scorer_version = require_math_verify()
    if "bm" in args.strategies and args.checkpoint is None:
        raise ValueError("--checkpoint is required for BM evaluation")
    if args.block_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens and block-length must be positive")
    if args.max_new_tokens % args.block_length:
        raise ValueError("max-new-tokens must be divisible by block-length")
    rows = read_jsonl(args.dataset_jsonl)
    if args.limit:
        rows = rows[: args.limit]
    tokenizer, model = load_nemotron(args.model, args.device)
    eos_id = int(model.config.eos_token_id)
    qdiffusion, selection_hook = (
        load_nemotron_guidance(
            model,
            args.checkpoint,
            num_candidates=args.num_candidates,
            energy_lambda=args.energy_lambda,
        )
        if "bm" in args.strategies
        else (None, None)
    )
    results = []
    for strategy in args.strategies:
        items_path, items, fingerprint = _partial_store(args, strategy)
        completed_indices = [int(item["index"]) for item in items]
        if len(completed_indices) != len(set(completed_indices)):
            raise ValueError(f"duplicate indices in {strategy} partial output")
        if any(index < 0 or index >= len(rows) for index in completed_indices):
            raise ValueError(f"out-of-range index in {strategy} partial output")
        completed = set(completed_indices)
        for index, row in enumerate(rows):
            if index in completed:
                continue
            inputs = prompt_ids(tokenizer, str(row["problem"]), args.device)
            torch.manual_seed(args.seed + index)
            started = time.perf_counter()
            if strategy == "native":
                with NativeGenerationSession(model) as generation:
                    output, nfe = native_generate(
                        generation,
                        inputs,
                        max_new_tokens=args.max_new_tokens,
                        block_length=args.block_length,
                        threshold=args.threshold,
                        eos_token_id=eos_id,
                    )
                selection_stats = []
            else:
                assert qdiffusion is not None
                assert selection_hook is not None
                with NativeGenerationSession(
                    model,
                    proposal_hook=selection_hook,
                    capture_hidden_states=True,
                ) as generation:
                    output, nfe = native_generate(
                        generation,
                        inputs,
                        max_new_tokens=args.max_new_tokens,
                        block_length=args.block_length,
                        threshold=args.threshold,
                        eos_token_id=eos_id,
                    )
                selection_stats = selection_hook.get_stats()
                selection_hook.stats.clear()
            final_text = tokenizer.decode(
                output[0, inputs.size(1) :], skip_special_tokens=True
            )
            item = {
                "index": index,
                "problem_id": row.get("problem_id"),
                "prediction": last_boxed_content(final_text),
                "gold": str(row["answer"]),
                "correct": bool(math_equivalent(final_text, str(row["answer"]))),
                "nfe": int(nfe),
                "num_tokens": int(output.size(1) - inputs.size(1)),
                "elapsed_seconds": time.perf_counter() - started,
                "final_text": final_text,
                "selection_stats": selection_stats,
            }
            _append_item(items_path, item)
            items.append(item)
            print(
                json.dumps(
                    {
                        key: item[key]
                        for key in ("index", "correct", "nfe", "num_tokens")
                    }
                ),
                flush=True,
            )
        summary = _summarize(strategy, items, fingerprint)
        summary["scorer"] = {
            "name": "math-verify",
            "version": scorer_version,
            "boxed_final_answer_only": True,
        }
        atomic_json(args.output_dir / f"{strategy}.results.json", summary)
        results.append({key: value for key, value in summary.items() if key != "items"})
    atomic_json(args.output_dir / "results.json", {"results": results})


if __name__ == "__main__":
    main()
