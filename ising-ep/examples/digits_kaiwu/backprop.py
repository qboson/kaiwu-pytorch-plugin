"""Backpropagation baseline on the same Digits split as the EP experiment.

The baseline is a multilayer perceptron with the EP network's hidden widths:
``64 -> hidden_sizes -> 10`` with ``tanh`` hidden units (the continuous
counterpart of a +-1 spin) and a softmax cross-entropy output. It is trained
with plain SGD through a ``DataLoader`` for the same number of epochs and the
same batch size as the EP run, so the comparison isolates the learning rule
rather than the training budget.

Every epoch records the mean training loss, training and test accuracy, and the
wall time of the training pass. As in the EP experiment, everything random is
derived from the experiment seed, so a given ``(seed, configuration)`` pair
reproduces bit for bit on the same platform.

Run with ``--help`` for the option list, or see ``README.md`` in this directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from examples.mnist_kaiwu.artifacts import utc_now, write_json
from examples.mnist_kaiwu.experiment import derived_int
from examples.mnist_kaiwu.metrics import classification_metrics, population_stats

from .data import N_CLASSES, N_INPUTS, SPLIT_SEED, TEST_FRACTION, DigitsDataset, load_digits_splits


@dataclass(frozen=True)
class BackpropConfig:
    """Settings of one baseline run, recorded to ``config.json``.

    Attributes:
        seeds: Experiment seeds to run.
        hidden_sizes: Width of each hidden layer, matching the EP network.
        epochs: Passes over the training split.
        batch_size: Samples per SGD step.
        lr: SGD learning rate.
        momentum: SGD momentum.
        split_seed: Seed of the stratified train/test partition.
        test_fraction: Fraction of all images held out for testing.
    """

    seeds: tuple[int, ...] = (0, 1, 2)
    hidden_sizes: tuple[int, ...] = (64,)
    epochs: int = 5
    batch_size: int = 10
    lr: float = 0.05
    momentum: float = 0.0
    split_seed: int = SPLIT_SEED
    test_fraction: float = TEST_FRACTION

    def to_json(self) -> dict[str, Any]:
        """Render the configuration as a JSON-serialisable mapping.

        Returns:
            Mapping with the model and training settings.
        """
        return {
            "created_at_utc": utc_now(),
            "seeds": list(self.seeds),
            "model": {
                "kind": "mlp_tanh_softmax",
                "layer_sizes": [N_INPUTS, *self.hidden_sizes, N_CLASSES],
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "optimizer": "sgd",
                "lr": self.lr,
                "momentum": self.momentum,
                "loss": "cross_entropy",
            },
        }


def build_mlp(hidden_sizes: Sequence[int], seed: int) -> nn.Sequential:
    """Build the baseline network with seeded PyTorch default initialization.

    Args:
        hidden_sizes: Width of each hidden layer.
        seed: Seed for the parameter initialization.

    Returns:
        ``Linear -> Tanh`` per hidden layer, followed by a ``Linear`` readout.
    """
    widths = [N_INPUTS, *(int(width) for width in hidden_sizes)]
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        layers: list[nn.Module] = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            layers.extend([nn.Linear(fan_in, fan_out), nn.Tanh()])
        layers.append(nn.Linear(widths[-1], N_CLASSES))
        model = nn.Sequential(*layers)
    finally:
        # Leave the global generator as we found it.
        torch.random.set_rng_state(generator_state)
    return model


def predict(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Predict class indices without tracking gradients.

    Args:
        model: Baseline network.
        images: Image batch of shape ``(n, 64)``.

    Returns:
        Predicted class indices of shape ``(n,)``.
    """
    model.eval()
    with torch.no_grad():
        return model(images).argmax(dim=1)


def accuracy(model: nn.Module, dataset: DigitsDataset) -> float:
    """Score a model on a whole dataset.

    Args:
        model: Baseline network.
        dataset: Images and labels to score.

    Returns:
        Fraction of correctly classified samples.
    """
    return float((predict(model, dataset.images) == dataset.labels).double().mean().item())


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
) -> float:
    """Run one SGD pass over ``loader``.

    Args:
        model: Network being trained.
        optimizer: Optimizer over ``model``'s parameters.
        loader: Training batches.

    Returns:
        Sample-weighted mean cross-entropy over the epoch.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    count = 0
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        (loss / labels.shape[0]).backward()
        optimizer.step()
        total += float(loss.item())
        count += int(labels.shape[0])
    return total / max(count, 1)


def run_seed(
    config: BackpropConfig,
    train: DigitsDataset,
    test: DigitsDataset,
    seed: int,
    seed_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate one seed, writing its artifacts under ``seed_dir``.

    Args:
        config: Run configuration shared by all seeds.
        train: Training dataset.
        test: Test dataset, scored after every epoch but never trained on.
        seed: Experiment seed.
        seed_dir: Directory receiving ``history.json``, ``model.pt``,
            ``test/metrics.json`` and ``status.json``.

    Returns:
        Status mapping with final train and test accuracy and mean epoch time.
    """
    seed_dir = Path(seed_dir)
    model = build_mlp(config.hidden_sizes, derived_int(seed, "bp-model"))
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    generator = torch.Generator().manual_seed(derived_int(seed, "bp-shuffle"))
    loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator)

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        started = time.perf_counter()
        loss = train_epoch(model, optimizer, loader)
        seconds = time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "train_accuracy": accuracy(model, train),
                "test_accuracy": accuracy(model, test),
                "epoch_seconds": seconds,
            }
        )

    predictions = predict(model, test.images).tolist()
    metrics = classification_metrics(
        test.labels.numpy(), predictions, [False] * len(predictions), n_classes=N_CLASSES
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), seed_dir / "model.pt")
    write_json(seed_dir / "history.json", {"epochs": history})
    write_json(seed_dir / "test" / "metrics.json", metrics)
    status = {
        "seed": int(seed),
        "train_accuracy": history[-1]["train_accuracy"],
        "test_accuracy": history[-1]["test_accuracy"],
        "final_train_loss": history[-1]["train_loss"],
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
    }
    write_json(seed_dir / "status.json", status)
    return status


def aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise per-seed baseline results.

    Args:
        results: Status mappings returned by :func:`run_seed`.

    Returns:
        Mapping with cross-seed statistics of train and test accuracy and of
        the mean epoch time.
    """
    return {
        "n_seeds": len(results),
        "seeds": [int(item["seed"]) for item in results],
        "train_accuracy": population_stats([float(item["train_accuracy"]) for item in results]),
        "test_accuracy": population_stats([float(item["test_accuracy"]) for item in results]),
        "mean_epoch_seconds": population_stats(
            [float(item["mean_epoch_seconds"]) for item in results]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Parser accepting the output directory, seeds, network widths, training
        settings and the split parameters.
    """
    parser = argparse.ArgumentParser(
        description="Train the backpropagation MLP baseline on the Digits split."
    )
    parser.add_argument("--output", required=True, help="Directory receiving run artifacts")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated experiment seeds")
    parser.add_argument(
        "--hidden-sizes", default="64", help="Comma-separated hidden widths, as in the EP run"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    return parser


def config_from_args(args: argparse.Namespace) -> BackpropConfig:
    """Translate parsed arguments into a :class:`BackpropConfig`.

    Args:
        args: Namespace produced by :func:`build_parser`.

    Returns:
        The run configuration.
    """
    return BackpropConfig(
        seeds=tuple(int(part) for part in args.seeds.split(",") if part.strip()),
        hidden_sizes=tuple(int(part) for part in args.hidden_sizes.split(",") if part.strip()),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        momentum=float(args.momentum),
        split_seed=int(args.split_seed),
        test_fraction=float(args.test_fraction),
    )


def main(argv: list[str] | None = None) -> int:
    """Run every configured seed and write the cross-seed summary.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success.
    """
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    # Keep BLAS from picking a thread count that differs between machines.
    torch.set_num_threads(1)
    splits = load_digits_splits(split_seed=config.split_seed, test_fraction=config.test_fraction)
    train = DigitsDataset(splits.train)
    test = DigitsDataset(splits.test)

    output_dir = Path(args.output)
    write_json(output_dir / "config.json", {**config.to_json(), "data": splits.to_json()})

    results = []
    for seed in config.seeds:
        status = run_seed(config, train, test, int(seed), output_dir / "seeds" / str(int(seed)))
        results.append(status)
        print(
            f"seed {seed}: train={status['train_accuracy']:.4f} "
            f"test={status['test_accuracy']:.4f} epoch_s={status['mean_epoch_seconds']:.3f}"
        )

    summary = aggregate(results)
    write_json(output_dir / "summary.json", summary)
    test_stats = summary["test_accuracy"]
    print(
        f"{summary['n_seeds']} seed(s): test accuracy mean={test_stats['mean']:.4f} "
        f"std={test_stats['std']:.4f}"
    )
    print("wrote", output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
