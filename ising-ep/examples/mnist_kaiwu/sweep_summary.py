"""Compare tuning runs under ``runs/tune`` in one table.

Every run directory holds ``config.json`` and per-seed artifacts written with
``--validation-per-class``. The table lists what differs from the reference
configuration, the final validation accuracy, the tie rate, the per-epoch
validation curve and the wall time, per run (mean over its seeds).

Usage::

    python -m examples.mnist_kaiwu.sweep_summary --root runs/tune
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    """Read a JSON file.

    Args:
        path: File to read.

    Returns:
        The decoded JSON value.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Flatten a nested mapping into dotted keys.

    Args:
        prefix: Key prefix accumulated so far.
        value: Value to flatten.
        out: Mapping receiving the flat entries.
    """
    if isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, inner, out)
    else:
        out[prefix] = value


def flat_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten a ``config.json`` payload, dropping volatile fields.

    Args:
        config: Decoded ``config.json``.

    Returns:
        Dotted-key mapping without the creation timestamp or the seed list.
    """
    out: dict[str, Any] = {}
    _flatten("", config, out)
    for key in ("created_at_utc", "seeds"):
        out.pop(key, None)
    return out


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Collect the headline numbers of one tuning run.

    Args:
        run_dir: Directory holding ``config.json`` and ``seeds/``.

    Returns:
        Mapping with the flat config, mean validation accuracy and tie rate,
        the mean per-epoch accuracy curve, and the mean elapsed seconds.
    """
    config = flat_config(_load(run_dir / "config.json"))
    accuracies: list[float] = []
    ties: list[float] = []
    elapsed: list[float] = []
    curves: list[list[float]] = []
    for seed_dir in sorted((run_dir / "seeds").glob("*")):
        status_path = seed_dir / "status.json"
        if not status_path.is_file():
            continue
        status = _load(status_path)
        accuracies.append(float(status["accuracy"]))
        ties.append(float(status["tie_rate"]))
        if "elapsed_seconds" in status:
            elapsed.append(float(status["elapsed_seconds"]))
        files = sorted(
            (seed_dir / "epochs").glob("*/validation.json"),
            key=lambda path: int(path.parent.name),
        )
        curves.append([float(_load(path)["accuracy"]) for path in files])

    def mean(values: list[float]) -> float | None:
        """Return the mean, or ``None`` when empty.

        Args:
            values: Numbers to average.

        Returns:
            The mean or ``None``.
        """
        return sum(values) / len(values) if values else None

    length = min((len(curve) for curve in curves), default=0)
    mean_curve = [mean([curve[index] for curve in curves]) for index in range(length)]
    return {
        "name": run_dir.name,
        "config": config,
        "n_seeds": len(accuracies),
        "accuracy": mean(accuracies),
        "tie_rate": mean(ties),
        "curve": mean_curve,
        "elapsed_seconds": mean(elapsed),
    }


def config_diff(config: dict[str, Any], reference: dict[str, Any]) -> str:
    """Describe how a configuration differs from a reference.

    Args:
        config: Flat configuration of a run.
        reference: Flat configuration to compare against.

    Returns:
        Comma-separated ``key=value`` pairs, or ``-`` when identical.
    """
    parts = [
        f"{key}={value}"
        for key, value in config.items()
        if reference.get(key) != value
    ]
    return ", ".join(parts) if parts else "-"


def format_table(rows: list[dict[str, Any]], reference: dict[str, Any]) -> str:
    """Render the comparison table as plain text.

    Args:
        rows: Output of :func:`summarize_run` for each run.
        reference: Flat configuration the differences are measured against.

    Returns:
        A multi-line, markdown-style table.
    """
    lines = [
        "| run | diff vs reference | seeds | val acc | tie | curve | sec |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        curve = " ".join(f"{value:.2f}" for value in row["curve"])
        seconds = "-" if row["elapsed_seconds"] is None else f"{row['elapsed_seconds']:.0f}"
        accuracy = "-" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
        tie = "-" if row["tie_rate"] is None else f"{row['tie_rate']:.3f}"
        lines.append(
            f"| {row['name']} | {config_diff(row['config'], reference)} | {row['n_seeds']} "
            f"| {accuracy} | {tie} | {curve} | {seconds} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the comparison table for every run under ``--root``.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when no run is found.
    """
    parser = argparse.ArgumentParser(description="Compare tuning runs.")
    parser.add_argument("--root", default="runs/tune", help="Directory of run directories")
    parser.add_argument(
        "--reference",
        default=None,
        help="Run name whose config is the baseline for the diff column",
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    runs = [path for path in sorted(root.glob("*")) if (path / "config.json").is_file()]
    if not runs:
        print("no runs under", root)
        return 1
    rows = [summarize_run(path) for path in runs]
    chosen = next((row for row in rows if row["name"] == args.reference), rows[0])
    print(format_table(rows, chosen["config"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
