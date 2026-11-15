"""Command-line driver for the sklearn Digits Ising EP experiment.

The network is ``64 -> hidden_sizes -> 40``: 64 clamped pixels, the hidden
spin layers, and 10 classes x 4 output spins. Training, evaluation, artifacts
and the cross-seed summary are shared with the MNIST/100 example
(:mod:`examples.mnist_kaiwu.experiment`); only the data differ. The input
width does not change the solver problem, which holds the hidden and output
spins only, so a Digits solve costs the same as a MNIST/100 solve of the same
network.

``config.json`` gains a ``data`` block recording the split seed, fractions and
split sizes, so the exact partition can be rebuilt from the file alone.

Run with ``--help`` for the full option list, or see ``README.md`` in this
directory for worked commands, including the backpropagation baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from ising_ep.kaiwu_solver import KaiwuSolver, init_license

from examples.mnist_kaiwu.artifacts import load_dotenv, write_json
from examples.mnist_kaiwu.experiment import (
    RunConfig,
    add_run_arguments,
    config_from_args as shared_config_from_args,
    run_seeds,
)

from .data import N_CLASSES, N_INPUTS, SPLIT_SEED, TEST_FRACTION, load_digits_splits


def build_parser() -> argparse.ArgumentParser:
    """Build the Digits command-line parser.

    Returns:
        Parser accepting the split parameters plus every shared model,
        training and annealing option.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Ising EP network on sklearn Digits via Kaiwu."
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=SPLIT_SEED,
        help="Seed of the stratified train/test partition, independent of --seeds",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=TEST_FRACTION,
        help="Fraction of all images held out for testing",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Fraction of the training part held out for validation; 0 disables it",
    )
    return add_run_arguments(parser)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Translate parsed arguments into a :class:`RunConfig` for 64 inputs.

    Args:
        args: Namespace produced by :func:`build_parser`.

    Returns:
        The shared run configuration with ``n_in`` set to the Digits width.
    """
    return dataclasses.replace(
        shared_config_from_args(args), n_in=N_INPUTS, n_classes=N_CLASSES
    )


def main(argv: list[str] | None = None) -> int:
    """Run every configured seed and write the cross-seed summary.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success.

    Raises:
        ValueError: If validation is requested without a validation split.
    """
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    splits = load_digits_splits(
        split_seed=args.split_seed,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
    )
    if (config.eval_every_epoch or config.skip_test) and splits.validation is None:
        raise ValueError("--eval-every-epoch and --skip-test need --validation-fraction > 0")

    # Credentials are optional: the classical annealer runs without them, but a
    # licensed session is required for the hosted backends.
    credentials = load_dotenv()
    if credentials:
        init_license(*credentials)

    output_dir = Path(args.output)
    write_json(output_dir / "config.json", {**config.to_json(), "data": splits.to_json()})
    run_seeds(
        config,
        KaiwuSolver(config.anneal),
        (splits.train.images, splits.train.labels),
        (splits.test.images, splits.test.labels),
        output_dir,
        validation=(
            None
            if splits.validation is None
            else (splits.validation.images, splits.validation.labels)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
