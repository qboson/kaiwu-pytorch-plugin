"""Command-line driver for the MNIST/100 Ising EP experiment.

One invocation trains and evaluates every requested seed, writes per-seed
artifacts under ``<output>/seeds/<seed>/`` and a cross-seed ``summary.json`` at
the top of the output directory.

Everything random is derived from the seed with :func:`derived_int`, so a given
``(seed, configuration)`` pair reproduces bit for bit: parameter initialization,
the per-epoch shuffle, each sample's paired anneal seed, and the evaluation
seeds.

Run with ``--help`` for the full option list, or see ``README.md`` in this
directory for worked commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from ising_ep.kaiwu_solver import AnnealSettings, KaiwuSolver, init_license
from ising_ep.model import IsingEPModel

from .artifacts import load_dotenv, utc_now, write_json, write_npz
from .data import load_mnist100, load_validation_split
from .evaluate import evaluate_free_only
from .metrics import l2_distance, learning_evidence, population_stats, relative_change
from .train import train_epoch


def _rates_of_length(
    values: Sequence[float] | None,
    default: float,
    length: int,
    name: str,
) -> tuple[float, ...]:
    """Repeat ``default`` or check that ``values`` has exactly ``length`` entries.

    Args:
        values: Explicit per-layer rates, or ``None`` to fill with ``default``.
        default: Value used when ``values`` is ``None``.
        length: Required number of rates.
        name: Field name used in the error message.

    Returns:
        A tuple of ``length`` floats.

    Raises:
        ValueError: If ``values`` is set and its length is not ``length``.
    """
    if values is None:
        return (float(default),) * int(length)
    if len(values) != int(length):
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return tuple(float(value) for value in values)


def _parse_optional_floats(raw: str | None) -> tuple[float, ...] | None:
    """Parse a comma-separated float list, treating a blank string as unset.

    Args:
        raw: Comma-separated numbers, or ``None``/empty.

    Returns:
        The parsed tuple, or ``None`` when the option was omitted.
    """
    if raw is None or not str(raw).strip():
        return None
    return tuple(float(part) for part in str(raw).split(",") if part.strip())


def derived_int(seed: int, domain: str, *parts: int) -> int:
    """Derive a reproducible 31-bit integer from a seed and a label.

    Hashing the seed together with a domain string keeps the different random
    streams independent: the model initialization, the epoch shuffle and each
    sample's anneal cannot accidentally share a value just because their
    indices coincide. The hash only splits those streams. It is not stored and
    it is not a provenance record.

    Args:
        seed: Experiment seed for this run.
        domain: Name of the random stream, for example ``"model"`` or ``"pair"``.
        *parts: Extra integers identifying a position within the stream, such as
            an epoch and a sample index.

    Returns:
        An integer in ``[0, 2**31)``.
    """
    material = "|".join((str(int(seed)), domain, *(str(int(part)) for part in parts)))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce a run, recorded to ``config.json``.

    Attributes:
        seeds: Experiment seeds to run, each producing its own directory.
        hidden_sizes: Width of each hidden spin layer, in order.
        epochs: Passes over the training split.
        batch_size: Samples sharing one parameter state before each update.
        lr: Default SGD learning rate, used when a per-layer list is omitted.
        lr_weights: Per-connection weight learning rates, input first. ``None``
            repeats :attr:`lr` once per connection.
        lr_biases: Per-layer bias learning rates. ``None`` repeats :attr:`lr`.
        init_gains: Per-connection init scales. ``None`` means every gain is 1.
        weight_decay: L2 decay applied to the weight matrices only.
        beta: Nudging coefficient, strictly positive.
        anneal: Annealing schedule shared by every solve.
        n_in: Input pixels, 784 for MNIST.
        n_classes: Number of digit classes.
        spins_per_class: Output spins allocated to each class.
        validation_per_class: Validation images per class held out from the
            training file. ``0`` disables the validation split.
        eval_every_epoch: Evaluate on the validation split after every epoch.
        skip_test: Never touch the test split; the reported accuracy is then the
            final-epoch validation accuracy.
    """

    seeds: tuple[int, ...] = (0, 1, 2)
    hidden_sizes: tuple[int, ...] = (64,)
    epochs: int = 5
    batch_size: int = 10
    lr: float = 0.01
    lr_weights: tuple[float, ...] | None = None
    lr_biases: tuple[float, ...] | None = None
    init_gains: tuple[float, ...] | None = None
    weight_decay: float = 0.0
    beta: float = 1.0
    anneal: AnnealSettings = field(default_factory=AnnealSettings)
    n_in: int = 784
    n_classes: int = 10
    spins_per_class: int = 4
    validation_per_class: int = 0
    eval_every_epoch: bool = False
    skip_test: bool = False

    @property
    def n_connections(self) -> int:
        """Number of weight matrices, one per adjacent pair of layers.

        Returns:
            ``len(hidden_sizes) + 1``.
        """
        return len(self.hidden_sizes) + 1

    def resolved_lr_weights(self) -> tuple[float, ...]:
        """Return one weight learning rate per connection.

        Returns:
            :attr:`lr_weights` when set, otherwise :attr:`lr` repeated.
        """
        return _rates_of_length(self.lr_weights, self.lr, self.n_connections, "lr_weights")

    def resolved_lr_biases(self) -> tuple[float, ...]:
        """Return one bias learning rate per spin layer.

        Returns:
            :attr:`lr_biases` when set, otherwise :attr:`lr` repeated.
        """
        return _rates_of_length(self.lr_biases, self.lr, self.n_connections, "lr_biases")

    def resolved_init_gains(self) -> tuple[float, ...]:
        """Return one init scale per connection.

        Returns:
            :attr:`init_gains` when set, otherwise a tuple of ones.
        """
        if self.init_gains is None:
            return (1.0,) * self.n_connections
        return _rates_of_length(self.init_gains, 1.0, self.n_connections, "init_gains")

    def to_json(self) -> dict[str, Any]:
        """Render the configuration as a JSON-serialisable mapping.

        Returns:
            Nested mapping with the model, training and annealing settings.
        """
        payload = self._base_json()
        if self.validation_per_class or self.eval_every_epoch or self.skip_test:
            # Recorded only when used, so default runs keep their exact layout.
            payload["validation"] = {
                "validation_per_class": self.validation_per_class,
                "eval_every_epoch": self.eval_every_epoch,
                "skip_test": self.skip_test,
            }
        return payload

    def _base_json(self) -> dict[str, Any]:
        """Render the settings shared by every run.

        Returns:
            Nested mapping with the model, training and annealing settings.
        """
        return {
            "created_at_utc": utc_now(),
            "seeds": list(self.seeds),
            "model": {
                "n_in": self.n_in,
                "hidden_sizes": list(self.hidden_sizes),
                "n_classes": self.n_classes,
                "spins_per_class": self.spins_per_class,
                "n_output": self.n_classes * self.spins_per_class,
                "layer_sizes": [
                    self.n_in,
                    *self.hidden_sizes,
                    self.n_classes * self.spins_per_class,
                ],
                "init_gains": list(self.resolved_init_gains()),
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "lr_weights": list(self.resolved_lr_weights()),
                "lr_biases": list(self.resolved_lr_biases()),
                "weight_decay": self.weight_decay,
                "beta": self.beta,
            },
            "anneal": {
                "initial_temperature": self.anneal.initial_temperature,
                "alpha": self.anneal.alpha,
                "cutoff_temperature": self.anneal.cutoff_temperature,
                "iterations_per_t": self.anneal.iterations_per_t,
                "candidate_count": self.anneal.candidate_count,
            },
        }


def build_optimizer(model: IsingEPModel, config: RunConfig) -> torch.optim.SGD:
    """Build SGD with one parameter group per weight matrix and per bias.

    Weight groups use :meth:`RunConfig.resolved_lr_weights` and
    :attr:`RunConfig.weight_decay`. Bias groups use
    :meth:`RunConfig.resolved_lr_biases` and no decay, matching the numpy
    reference that regularises couplings more strongly than biases.

    Args:
        model: Network whose parameters are grouped.
        config: Run configuration supplying the per-layer rates.

    Returns:
        An SGD optimizer with two groups per connection.

    Raises:
        ValueError: If a per-layer rate list has the wrong length.
    """
    lr_weights = config.resolved_lr_weights()
    lr_biases = config.resolved_lr_biases()
    groups: list[dict[str, Any]] = []
    for index, (weight, bias) in enumerate(zip(model.weights, model.biases)):
        groups.append(
            {"params": [weight], "lr": lr_weights[index], "weight_decay": config.weight_decay}
        )
        groups.append({"params": [bias], "lr": lr_biases[index], "weight_decay": 0.0})
    return torch.optim.SGD(groups, lr=config.lr)


def run_seed(
    config: RunConfig,
    solver: KaiwuSolver,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
    seed_dir: Path,
    *,
    validation_images: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train and evaluate one seed, writing its artifacts under ``seed_dir``.

    Test evaluation runs only after the final epoch, on the checkpoint written
    at that point, so no test image influences training. With
    ``config.eval_every_epoch`` the validation split is also scored after each
    epoch into ``epochs/<e>/validation.json``. With ``config.skip_test`` the
    test split is never read and the reported accuracy is the final-epoch
    validation accuracy.

    Args:
        config: Run configuration shared by all seeds.
        solver: Annealer bound to the shared schedule.
        train_images: Training images, one row per sample.
        train_labels: Training labels aligned with ``train_images``.
        test_images: Test images, one row per sample.
        test_labels: Test labels aligned with ``test_images``.
        seed: Experiment seed for this run.
        seed_dir: Directory receiving this seed's artifacts.
        validation_images: Validation images, required when validating.
        validation_labels: Validation labels aligned with ``validation_images``.

    Returns:
        Status mapping with accuracy, tie rate, per-class accuracy, learning
        evidence and parameter-change summaries.

    Raises:
        ValueError: If validation is requested without a validation split.
    """
    validating = config.eval_every_epoch or config.skip_test
    if validating and (validation_images is None or validation_labels is None):
        raise ValueError("eval_every_epoch and skip_test need a validation split")
    seed_dir = Path(seed_dir)
    model = IsingEPModel(
        n_in=config.n_in,
        hidden_sizes=config.hidden_sizes,
        n_classes=config.n_classes,
        spins_per_class=config.spins_per_class,
        seed=derived_int(seed, "model"),
        init_gains=config.resolved_init_gains(),
    )
    initial = model.parameter_snapshot()
    write_npz(seed_dir / "initial-model.npz", initial)

    optimizer = build_optimizer(model, config)
    hidden_nonzero_rates: list[float] = []
    started = time.perf_counter()
    validation_calls = 0
    validation_metrics: dict[str, Any] = {}
    for epoch in range(config.epochs):
        shuffle = np.random.default_rng(derived_int(seed, "shuffle", epoch))
        order = shuffle.permutation(int(train_images.shape[0]))

        def pair_seed_fn(sample_index: int, epoch_index: int = epoch) -> int:
            """Return the paired anneal seed for one sample in this epoch.

            Args:
                sample_index: Index of the sample being solved.
                epoch_index: Bound to the current epoch at definition time so
                    the closure does not capture the loop variable.

            Returns:
                The seed shared by that sample's free and nudged solves.
            """
            return derived_int(seed, "pair", epoch_index, sample_index)

        stats = train_epoch(
            model,
            optimizer,
            solver,
            train_images,
            train_labels,
            order,
            beta=config.beta,
            batch_size=config.batch_size,
            pair_seed_fn=pair_seed_fn,
        )
        # A hidden state that never flips means the nudge produced no signal.
        hidden_nonzero_rates.append(1.0 - float(stats["hidden_zero_update"]))
        epoch_dir = seed_dir / "epochs" / str(epoch)
        write_npz(epoch_dir / "model.npz", model.parameter_snapshot())
        write_json(epoch_dir / "dynamics.json", stats)
        last_epoch = epoch == config.epochs - 1
        if config.eval_every_epoch or (config.skip_test and last_epoch):
            validation_metrics, _ = evaluate_free_only(
                model,
                solver,
                validation_images,
                validation_labels,
                eval_seed=derived_int(seed, "validation", epoch),
            )
            write_json(epoch_dir / "validation.json", validation_metrics)
            validation_calls += int(len(validation_labels))

    final = model.parameter_snapshot()
    write_npz(seed_dir / "checkpoint.npz", final)

    if config.skip_test:
        metrics = validation_metrics
    else:
        metrics, records = evaluate_free_only(
            model,
            solver,
            test_images,
            test_labels,
            eval_seed=derived_int(seed, "eval"),
        )
        write_json(seed_dir / "test" / "metrics.json", metrics)
        write_json(seed_dir / "test" / "predictions.json", {"records": records})

    weight_keys = [key for key in initial if key.startswith("weights.")]
    status = {
        "seed": int(seed),
        "accuracy": metrics["accuracy"],
        "tie_rate": metrics["tie_rate"],
        "per_class": metrics["per_class"],
        "learning_evidence": learning_evidence(
            weight_changes=[l2_distance(initial[key], final[key]) for key in weight_keys],
            hidden_nonzero_rates=hidden_nonzero_rates,
        ),
        "param_change": {
            key: relative_change(initial[key], final[key]) for key in weight_keys
        },
        "solver_calls": {
            # Two calls per training sample per epoch, one per test image.
            "train_free": int(train_images.shape[0]) * config.epochs,
            "train_nudged": int(train_images.shape[0]) * config.epochs,
            "test_evaluation": 0 if config.skip_test else int(test_images.shape[0]),
        },
    }
    if validation_calls:
        status["solver_calls"]["validation_evaluation"] = validation_calls
        # Wall time is only recorded for tuning runs, so default artifacts stay
        # bit-for-bit reproducible.
        status["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    if config.skip_test:
        status["evaluated_on"] = "validation"
    write_json(seed_dir / "status.json", status)
    return status


def aggregate(results: Sequence[dict[str, Any]], n_classes: int = 10) -> dict[str, Any]:
    """Summarise per-seed results across seeds.

    Args:
        results: Status mappings returned by :func:`run_seed`.
        n_classes: Number of classes to report per-class statistics for.

    Returns:
        Mapping with the seed list, accuracy and tie-rate statistics, and
        per-class accuracy statistics.
    """
    per_class = [
        {
            "class": index,
            **population_stats(
                [
                    float(row["accuracy"])
                    for item in results
                    for row in item["per_class"]
                    if int(row["class"]) == index
                ]
            ),
        }
        for index in range(int(n_classes))
    ]
    return {
        "n_seeds": len(results),
        "seeds": [int(item["seed"]) for item in results],
        "accuracy": population_stats([float(item["accuracy"]) for item in results]),
        "tie_rate": population_stats([float(item["tie_rate"]) for item in results]),
        "per_class": per_class,
        "learning_evidence_passed": all(
            bool(item["learning_evidence"]["passed"]) for item in results
        ),
    }


def run_seeds(
    config: RunConfig,
    solver: KaiwuSolver,
    train: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    output_dir: Path,
    *,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run every configured seed and write the cross-seed ``summary.json``.

    The dataset only enters through the arrays, so every example shares this
    loop: MNIST/100 and sklearn Digits differ in their loaders alone.

    Args:
        config: Run configuration shared by all seeds.
        solver: Annealer bound to the shared schedule.
        train: Training images and labels.
        test: Test images and labels.
        output_dir: Directory receiving ``seeds/<seed>/`` and ``summary.json``.
        validation: Validation images and labels, if a validation split exists.

    Returns:
        The cross-seed summary written to ``summary.json``.
    """
    output_dir = Path(output_dir)
    results = []
    for seed in config.seeds:
        status = run_seed(
            config,
            solver,
            train[0],
            train[1],
            test[0],
            test[1],
            int(seed),
            output_dir / "seeds" / str(int(seed)),
            validation_images=None if validation is None else validation[0],
            validation_labels=None if validation is None else validation[1],
        )
        results.append(status)
        print(f"seed {seed}: accuracy={status['accuracy']:.4f} tie_rate={status['tie_rate']:.4f}")

    summary = aggregate(results, n_classes=config.n_classes)
    write_json(output_dir / "summary.json", summary)
    accuracy = summary["accuracy"]
    print(
        f"{summary['n_seeds']} seed(s): accuracy mean={accuracy['mean']:.4f} "
        f"std={accuracy['std']:.4f} min={accuracy['min']:.4f} max={accuracy['max']:.4f}"
    )
    print("wrote", output_dir / "summary.json")
    return summary


def add_run_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the options every EP example shares: model, training and annealing.

    Dataset-specific options (where the data lives, how it is split) are added
    by each example's own parser.

    Args:
        parser: Parser to extend in place.

    Returns:
        The same parser, for chaining.
    """
    parser.add_argument("--output", required=True, help="Directory receiving run artifacts")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated experiment seeds")
    parser.add_argument("--n-hidden", type=int, default=64, help="Width of a single hidden layer")
    parser.add_argument(
        "--hidden-sizes",
        default=None,
        help="Comma-separated hidden widths. Overrides --n-hidden when set",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--lr-weights",
        default=None,
        help="Comma-separated weight learning rates, one per connection",
    )
    parser.add_argument(
        "--lr-biases",
        default=None,
        help="Comma-separated bias learning rates, one per spin layer",
    )
    parser.add_argument(
        "--init-gains",
        default=None,
        help="Comma-separated init scales, one per connection; default 1",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 decay applied to weight matrices only",
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--initial-temperature", type=float, default=1.25)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--cutoff-temperature", type=float, default=0.001)
    parser.add_argument("--iterations-per-t", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument(
        "--eval-every-epoch",
        action="store_true",
        help="Score the validation split after every epoch",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Never evaluate the test split; report validation accuracy instead",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the MNIST/100 command-line parser.

    Returns:
        Parser accepting the output directory, data root, seeds, model and
        training hyperparameters, and the annealing schedule.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Ising EP network on MNIST/100 via Kaiwu."
    )
    parser.add_argument("--data-root", default="data", help="Directory holding MNIST/raw")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download missing MNIST archives instead of failing",
    )
    parser.add_argument(
        "--validation-per-class",
        type=int,
        default=0,
        help="Hold out this many extra training-file images per class for validation",
    )
    return add_run_arguments(parser)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Translate parsed arguments into a :class:`RunConfig`.

    Args:
        args: Namespace produced by :func:`build_parser`.

    Returns:
        The run configuration.
    """
    if args.hidden_sizes:
        hidden_sizes = tuple(int(part) for part in args.hidden_sizes.split(",") if part.strip())
    else:
        hidden_sizes = (int(args.n_hidden),)
    return RunConfig(
        seeds=tuple(int(part) for part in args.seeds.split(",")),
        hidden_sizes=hidden_sizes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_weights=_parse_optional_floats(args.lr_weights),
        lr_biases=_parse_optional_floats(args.lr_biases),
        init_gains=_parse_optional_floats(args.init_gains),
        weight_decay=float(args.weight_decay),
        beta=args.beta,
        anneal=AnnealSettings(
            initial_temperature=args.initial_temperature,
            alpha=args.alpha,
            cutoff_temperature=args.cutoff_temperature,
            iterations_per_t=args.iterations_per_t,
            candidate_count=args.candidate_count,
        ),
        # Only the MNIST parser defines this option.
        validation_per_class=int(getattr(args, "validation_per_class", 0)),
        eval_every_epoch=bool(args.eval_every_epoch),
        skip_test=bool(args.skip_test),
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

    # Credentials are optional: the classical annealer runs without them, but a
    # licensed session is required for the hosted backends.
    credentials = load_dotenv()
    if credentials:
        init_license(*credentials)

    output_dir = Path(args.output)
    write_json(output_dir / "config.json", config.to_json())

    train_split, test_split = load_mnist100(
        args.data_root, download_if_missing=args.download_data
    )
    validation_split = None
    if config.validation_per_class:
        validation_split = load_validation_split(
            args.data_root,
            config.validation_per_class,
            download_if_missing=args.download_data,
        )
    run_seeds(
        config,
        KaiwuSolver(config.anneal),
        (train_split.images, train_split.labels),
        (test_split.images, test_split.labels),
        output_dir,
        validation=(
            None
            if validation_split is None
            else (validation_split.images, validation_split.labels)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
