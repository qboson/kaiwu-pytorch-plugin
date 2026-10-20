"""Train the contextual KPP BM on private same-state outcome pairs."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .common.pairs import PAIR_SCHEMA_VERSION, load_pairs
from .common.runtime import atomic_json, file_identity
from .models.checkpoint import (
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from .models.energy import ContextualEnergyModel


class PairDataset(Dataset[dict[str, Any]]):
    """In-memory dataset over validated pair records.

    Attributes:
        records: Pair records returned unchanged per index.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Stacks a list of pair records into one batched dictionary.

    Args:
        records: Pair records sharing identical tensor fields.

    Returns:
        dict[str, Any]: Batch with stacked tensors plus per-row list fields.
    """
    tensor_keys = (
        "noisy_tokens",
        "hidden_states",
        "noisy_features",
        "positive_tokens",
        "negative_tokens",
        "positive_candidate_features",
        "negative_candidate_features",
        "transfer_mask",
    )
    batch: dict[str, Any] = {
        key: torch.stack([row[key] for row in records]) for key in tensor_keys
    }
    batch.update(
        {
            "positive_logprob": torch.tensor(
                [row["positive_logprob"] for row in records], dtype=torch.float32
            ),
            "negative_logprob": torch.tensor(
                [row["negative_logprob"] for row in records], dtype=torch.float32
            ),
            "problem_id": [row["problem_id"] for row in records],
            "negative_kind": [row["negative_kind"] for row in records],
            "block_index": [int(row["block_index"]) for row in records],
            "step_index": [int(row["step_index"]) for row in records],
        }
    )
    return batch


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Moves every tensor in the batch to the target device.

    Args:
        batch: Batch produced by ``collate_pairs``.
        device: Target torch device.

    Returns:
        dict[str, Any]: Batch with tensors moved and non-tensors unchanged.
    """
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def score_pair_batch(
    model: ContextualEnergyModel,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scores the positive and the negative candidate of each pair.

    Args:
        model: Contextual energy model used for scoring.
        batch: Batch produced by ``collate_pairs``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Positive and negative energies,
        each shaped ``[batch, 1]``.
    """
    positive = model.score_conditioned(
        batch["noisy_tokens"],
        batch["positive_tokens"],
        batch["transfer_mask"],
        hidden_states=batch["hidden_states"],
        noisy_features=batch["noisy_features"],
        candidate_features=batch["positive_candidate_features"],
    )
    negative = model.score_conditioned(
        batch["noisy_tokens"],
        batch["negative_tokens"],
        batch["transfer_mask"],
        hidden_states=batch["hidden_states"],
        noisy_features=batch["noisy_features"],
        candidate_features=batch["negative_candidate_features"],
    )
    return positive, negative


def pair_loss(
    positive_energy: torch.Tensor,
    negative_energy: torch.Tensor,
    *,
    margin: float = 0.5,
    nce_weight: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Outcome-pairwise margin loss plus a small energy-scale NCE regularizer.

    Args:
        positive_energy: Positive-pair energies shaped ``[batch, 1]``.
        negative_energy: Negative-pair energies shaped ``[batch, 1]``.
        margin: Minimum energy gap enforced between negatives and positives.
        nce_weight: Weight of the softplus NCE term that keeps energy
            magnitudes finite.

    Returns:
        Tuple ``(loss, ranking, nce)`` with the combined loss and its two
        unweighted components.

    Raises:
        ValueError: If ``margin`` is not positive or ``nce_weight`` is
            negative.
    """

    if margin <= 0 or nce_weight < 0:
        raise ValueError("margin must be positive and nce_weight non-negative")
    ranking = F.relu(margin + positive_energy - negative_energy).mean()
    nce = (F.softplus(positive_energy) + F.softplus(-negative_energy)).mean()
    return ranking + nce_weight * nce, ranking, nce


@torch.no_grad()
def evaluate_rows(
    model: ContextualEnergyModel,
    loader: Any,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Scores a loader of pair batches and collects per-pair rows.

    Args:
        model: Contextual energy model; switched to eval mode by this call.
        loader: DataLoader yielding collated pair batches.
        device: Device used for scoring.

    Returns:
        list[dict[str, Any]]: One row per pair with energies, log-probs, and
        identifiers.
    """
    model.eval()
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        positive, negative = score_pair_batch(model, batch)
        for index in range(positive.numel()):
            rows.append(
                {
                    "problem_id": batch["problem_id"][index],
                    "negative_kind": batch["negative_kind"][index],
                    "block_index": batch["block_index"][index],
                    "step_index": batch["step_index"][index],
                    "positive_energy": float(positive[index]),
                    "negative_energy": float(negative[index]),
                    "positive_logprob": float(batch["positive_logprob"][index]),
                    "negative_logprob": float(batch["negative_logprob"][index]),
                }
            )
    return rows


def energy_normalization(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Computes the mean/std used to normalize energies at inference.

    Args:
        rows: Per-pair rows produced by ``evaluate_rows``.

    Returns:
        dict[str, float]: ``mean`` and ``std`` over all pair energies.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("cannot normalize an empty evaluation")
    energies = torch.tensor(
        [
            value
            for row in rows
            for value in (row["positive_energy"], row["negative_energy"])
        ]
    )
    return {
        "mean": float(energies.mean()),
        "std": float(energies.std(unbiased=False).clamp_min(1e-6)),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    normalization: dict[str, float],
    energy_lambda: float = 2.0,
) -> dict[str, Any]:
    """Summarizes ranking quality of the trained scorer on pair rows.

    Args:
        rows: Per-pair rows produced by ``evaluate_rows``.
        normalization: Energy mean/std from ``energy_normalization``.
        energy_lambda: Residual weight applied to the normalized energy gap.

    Returns:
        dict[str, Any]: Aggregate ranking metrics, energy-gap quantiles, and
        per-negative-kind breakdowns.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("evaluation split contains no pairs")
    gaps = torch.tensor(
        [row["negative_energy"] - row["positive_energy"] for row in rows]
    )
    residual_wins = 0
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        proposal_gap = row["positive_logprob"] - row["negative_logprob"]
        energy_gap = (row["positive_energy"] - row["negative_energy"]) / normalization[
            "std"
        ]
        residual_wins += proposal_gap - energy_lambda * energy_gap > 0
        groups[row["negative_kind"]].append(row)
    return {
        "count": len(rows),
        "problem_count": len({row["problem_id"] for row in rows}),
        "ranking_accuracy": float((gaps > 0).float().mean()),
        "residual_ranking_accuracy": residual_wins / len(rows),
        "energy_gap_mean": float(gaps.mean()),
        "energy_gap_quantiles": {
            "q10": float(torch.quantile(gaps, 0.1)),
            "q50": float(torch.quantile(gaps, 0.5)),
            "q90": float(torch.quantile(gaps, 0.9)),
        },
        "by_negative_kind": {
            name: {
                "count": len(group),
                "ranking_accuracy": sum(
                    row["positive_energy"] < row["negative_energy"] for row in group
                )
                / len(group),
            }
            for name, group in sorted(groups.items())
        },
    }


# Training CLI and exact-resume state.


def parse_args() -> argparse.Namespace:
    """Parses training command-line arguments.

    Returns:
        argparse.Namespace: Parsed options; see ``--help`` and the README for
        the reference invocation of every flag.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bm-visible", type=int, default=512)
    parser.add_argument("--bm-hidden", type=int, default=256)
    parser.add_argument("--contextual-dim", type=int, default=1024)
    parser.add_argument("--contextual-layers", type=int, default=4)
    parser.add_argument("--contextual-heads", type=int, default=16)
    parser.add_argument("--contextual-ffn-dim", type=int, default=4096)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--sa-solutions", type=int, default=8)
    parser.add_argument("--sa-seed", type=int, default=20260855)
    parser.add_argument("--energy-num-lowest", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument(
        "--nce-weight",
        type=float,
        default=0.01,
        help="Energy-scale regularizer; the main objective remains outcome pairwise.",
    )
    parser.add_argument("--energy-lambda", type=float, default=2.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _git_revision(path: Path) -> str | None:
    """Returns the HEAD revision of the repository containing ``path``.

    Args:
        path: Path inside the repository to query.

    Returns:
        Full commit hash, or ``None`` when git is unavailable or the path
        is not inside a repository.
    """
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _compatibility_config(
    args: argparse.Namespace, first: dict[str, Any]
) -> dict[str, Any]:
    """Builds the compatibility metadata persisted with checkpoints.

    Args:
        args: Parsed CLI options.

        first: First pair record, used to derive feature dimensions.

    Returns:
        Dict describing schema, artifact identity, architecture, and
        optimization settings for resume compatibility checks.
    """
    return {
        "pair_schema_version": PAIR_SCHEMA_VERSION,
        "pair_artifact": file_identity(args.pairs),
        "proposal_hidden_dim": int(first["hidden_states"].shape[-1]),
        "candidate_feature_dim": int(first["positive_candidate_features"].shape[-1]),
        "bm_visible": args.bm_visible,
        "bm_hidden": args.bm_hidden,
        "contextual_dim": args.contextual_dim,
        "contextual_layers": args.contextual_layers,
        "contextual_heads": args.contextual_heads,
        "contextual_ffn_dim": args.contextual_ffn_dim,
        "max_sequence_length": args.max_sequence_length,
        "sa_solutions": args.sa_solutions,
        "sa_seed": args.sa_seed,
        "energy_num_lowest": args.energy_num_lowest,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "margin": args.margin,
        "nce_weight": args.nce_weight,
        "energy_lambda": args.energy_lambda,
        "seed": args.seed,
    }


def _build_loaders(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    generator: torch.Generator,
) -> dict[str, DataLoader[dict[str, Any]]]:
    """Builds one loader per problem-grouped split.

    Args:
        records: Validated pair records.

        args: Parsed CLI options supplying the batch size.

        generator: Random generator driving training-set shuffling.

    Returns:
        Mapping from split name to its ``DataLoader``.

    Raises:
        ValueError: If the train or val split is empty.
    """
    by_split = {
        split: [row for row in records if row["split"] == split]
        for split in ("train", "val")
    }
    if not by_split["train"] or not by_split["val"]:
        raise ValueError("problem-grouped train and val splits must both be non-empty")
    return {
        split: DataLoader(
            PairDataset(rows),
            batch_size=args.batch_size,
            shuffle=split == "train",
            collate_fn=collate_pairs,
            generator=generator if split == "train" else None,
        )
        for split, rows in by_split.items()
        if rows
    }


def _train_epoch(
    model: ContextualEnergyModel,
    loader: DataLoader,
    optimizer: AdamW,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Runs one optimization epoch over the training loader.

    Args:
        model: Contextual energy model trained in place.

        loader: Training loader of pair batches.

        optimizer: Optimizer stepped after each batch.

        device: Device batches are moved to.

        args: Parsed CLI options supplying loss and clipping settings.

    Returns:
        Dict with pair-count-weighted loss components and ranking accuracy.
    """
    model.train()
    totals = {key: 0.0 for key in ("loss", "ranking_loss", "nce_loss", "wins")}
    count = 0
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        positive, negative = score_pair_batch(model, batch)
        loss, ranking, nce = pair_loss(
            positive, negative, margin=args.margin, nce_weight=args.nce_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        size = positive.numel()
        count += size
        totals["loss"] += float(loss.detach()) * size
        totals["ranking_loss"] += float(ranking.detach()) * size
        totals["nce_loss"] += float(nce.detach()) * size
        totals["wins"] += int((positive < negative).sum())
    return {
        "loss": totals["loss"] / count,
        "ranking_loss": totals["ranking_loss"] / count,
        "nce_loss": totals["nce_loss"] / count,
        "ranking_accuracy": totals["wins"] / count,
    }


def main() -> None:
    """Runs contextual-energy training with exact resume support."""
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.energy_lambda < 0 or args.early_stopping_patience < 0:
        raise ValueError(
            "energy lambda and early-stopping patience must be non-negative"
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = load_pairs(args.pairs)
    data_generator = torch.Generator().manual_seed(args.seed + 1)
    loaders = _build_loaders(records, args, data_generator)
    compatibility = _compatibility_config(args, records[0])
    run_config = {
        "objective": "outcome_pairwise_plus_nce_energy_scale_regularizer",
        "compatibility": compatibility,
        "epochs_requested": args.epochs,
        "split_pair_counts": {
            split: sum(row["split"] == split for row in records)
            for split in loaders
        },
        "split_problem_counts": {
            split: len({row["problem_id"] for row in records if row["split"] == split})
            for split in loaders
        },
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "source_revision": _git_revision(Path(__file__).resolve().parents[3]),
        "benchmark_test_used_for_selection": False,
        "resume_events": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    config_path = args.output_dir / "run_config.json"
    device = torch.device(args.device)

    start_epoch = 1
    best_metric = (-1.0, -float("inf"))
    best_epoch: int | None = None
    without_improvement = 0
    history: list[dict[str, Any]] = []
    training_state: dict[str, Any] | None = None
    if last_path.exists():
        if not args.resume:
            raise FileExistsError(
                "last.pt exists; pass --resume or use a new output dir"
            )
        model, resumed = load_checkpoint(last_path, device=device)
        previous_run_config = resumed.get("run_config")
        if not isinstance(previous_run_config, dict):
            raise ValueError("last.pt has no valid run_config")
        previous = previous_run_config["compatibility"]
        if previous != compatibility:
            raise ValueError("resume configuration does not match last.pt")
        loaded_training_state = resumed.get("training_state")
        if not isinstance(loaded_training_state, dict):
            raise ValueError("last.pt has no resumable training_state")
        training_state = loaded_training_state
        start_epoch = int(resumed["epoch"]) + 1
        best_metric = tuple(training_state["best_metric"])
        best_epoch = training_state["best_epoch"]
        without_improvement = int(training_state["without_improvement"])
        history = list(training_state["history"])
        data_generator.set_state(training_state["data_generator_state"])
        torch.set_rng_state(training_state["torch_rng_state"])
        if torch.cuda.is_available() and training_state.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(training_state["cuda_rng_states"])
        run_config = dict(previous_run_config)
        run_config["epochs_requested"] = args.epochs
        run_config.setdefault("resume_events", []).append(
            {
                "from_epoch": int(resumed["epoch"]),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "source_revision": _git_revision(Path(__file__).resolve().parents[3]),
            }
        )
    else:
        first = records[0]
        model = ContextualEnergyModel(
            proposal_hidden_dim=int(first["hidden_states"].shape[-1]),
            candidate_feature_dim=int(first["positive_candidate_features"].shape[-1]),
            bm_num_visible=args.bm_visible,
            bm_num_hidden=args.bm_hidden,
            contextual_dim=args.contextual_dim,
            contextual_layers=args.contextual_layers,
            contextual_heads=args.contextual_heads,
            contextual_ffn_dim=args.contextual_ffn_dim,
            max_sequence_length=args.max_sequence_length,
            sa_num_solutions=args.sa_solutions,
            sa_seed=args.sa_seed,
            energy_num_lowest=args.energy_num_lowest,
            device=device,
        )

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    if training_state is not None:
        optimizer.load_state_dict(training_state["optimizer_state"])
        scheduler.load_state_dict(training_state["scheduler_state"])
    atomic_json(config_path, run_config)

    for epoch in range(start_epoch, args.epochs + 1):
        train_update = _train_epoch(
            model, loaders["train"], optimizer, device=device, args=args
        )
        train_rows = evaluate_rows(
            model,
            loaders["train"],
            device=device,
        )
        normalization = energy_normalization(train_rows)
        val_rows = evaluate_rows(
            model,
            loaders["val"],
            device=device,
        )
        train_metrics = summarize_rows(
            train_rows,
            normalization=normalization,
            energy_lambda=args.energy_lambda,
        )
        val_metrics = summarize_rows(
            val_rows,
            normalization=normalization,
            energy_lambda=args.energy_lambda,
        )
        scheduler.step(val_metrics["ranking_accuracy"])
        epoch_summary = {
            "epoch": epoch,
            "train_update": train_update,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary), flush=True)

        metric = (
            val_metrics["ranking_accuracy"],
            val_metrics["energy_gap_mean"],
        )
        improved = metric > best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
            without_improvement = 0
        else:
            without_improvement += 1
        state = {
            "best_metric": list(best_metric),
            "best_epoch": best_epoch,
            "without_improvement": without_improvement,
            "history": history,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "data_generator_state": data_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        }
        payload = checkpoint_payload(
            model,
            epoch=epoch,
            energy_mean=normalization["mean"],
            energy_std=normalization["std"],
            energy_lambda=args.energy_lambda,
            metrics={"train": train_metrics, "val": val_metrics},
            run_config=run_config,
            training_state=state,
        )
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(best_path, payload)
        if (
            args.early_stopping_patience
            and without_improvement >= args.early_stopping_patience
        ):
            break

    summary = {
        "best_epoch": best_epoch,
        "best_metric": list(best_metric),
        "history": history,
        "benchmark_test_used_for_selection": False,
    }
    atomic_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
