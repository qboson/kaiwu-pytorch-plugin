"""Tests for the Digits EP driver and the run loop it shares with MNIST/100."""

# Test functions document intent in their summary. RandomSolver mirrors the
# KaiwuSolver signature, so some arguments are deliberately unused.
# pylint: disable=missing-param-doc,missing-return-doc,unused-argument

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.digits_kaiwu.data import load_digits_splits
from examples.digits_kaiwu.experiment import build_parser, config_from_args, main
from examples.mnist_kaiwu.experiment import build_parser as mnist_parser
from examples.mnist_kaiwu.experiment import run_seeds
from ising_ep.kaiwu_solver import AnnealSettings

from tests.conftest import PRIMARY_ANNEAL

N_HIDDEN = 4
N_OUTPUT = 40


class RandomSolver:
    """Returns a seed-dependent spin state of the Digits network's width."""

    def solve(
        self,
        Q: Any,
        *,
        seed: int,
        init_solution: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one candidate row whose auxiliary spin is ``+1``."""
        width = int(np.asarray(Q).shape[0])
        state = np.random.default_rng(int(seed)).choice([-1.0, 1.0], size=width)
        return np.concatenate([state, [1.0]]).reshape(1, -1)


def _config(*extra: str) -> Any:
    """Parse a tiny Digits configuration."""
    return config_from_args(
        build_parser().parse_args(
            ["--output", "x", "--seeds", "0", "--n-hidden", str(N_HIDDEN), "--epochs", "1", *extra]
        )
    )


def test_config_has_64_inputs_and_the_shared_primary_defaults() -> None:
    """Only the input width differs from the MNIST/100 defaults."""
    config = config_from_args(build_parser().parse_args(["--output", "x"]))

    assert config.n_in == 64
    assert config.hidden_sizes == (64,)
    assert (config.epochs, config.batch_size, config.lr, config.beta) == (5, 10, 0.01, 1.0)
    assert config.anneal == PRIMARY_ANNEAL
    assert config.to_json()["model"]["layer_sizes"] == [64, 64, 40]


def test_digits_parser_has_split_options_and_no_mnist_data_root() -> None:
    """Dataset options belong to each example; the run options are shared."""
    args = build_parser().parse_args(["--output", "x"])

    assert (args.split_seed, args.test_fraction, args.validation_fraction) == (0, 0.2, 0.0)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--output", "x", "--data-root", "data"])
    assert mnist_parser().parse_args(["--output", "x"]).data_root == "data"


def test_shared_loop_trains_every_seed_on_digits(tmp_path: Path) -> None:
    """``run_seeds`` writes one tree per seed plus the summary, at 64 inputs."""
    config = dataclasses.replace(
        _config("--seeds", "0,1"), anneal=AnnealSettings(candidate_count=1)
    )
    splits = load_digits_splits()
    take = slice(0, 20)

    summary = run_seeds(
        config,
        RandomSolver(),
        (splits.train.images[take], splits.train.labels[take]),
        (splits.test.images[take], splits.test.labels[take]),
        tmp_path,
    )

    assert summary["seeds"] == [0, 1]
    assert (tmp_path / "summary.json").is_file()
    with np.load(tmp_path / "seeds" / "1" / "checkpoint.npz") as archive:
        shapes = {key: tuple(archive[key].shape) for key in archive.files}
    assert shapes["weights.0"] == (64, N_HIDDEN)
    assert shapes["weights.1"] == (N_HIDDEN, N_OUTPUT)
    status = json.loads((tmp_path / "seeds" / "0" / "status.json").read_text(encoding="utf-8"))
    assert status["solver_calls"] == {"train_free": 20, "train_nudged": 20, "test_evaluation": 20}


@pytest.mark.parametrize("flag", ["--skip-test", "--eval-every-epoch"])
def test_validation_flags_require_a_validation_fraction(tmp_path: Path, flag: str) -> None:
    """Validation-only runs fail fast instead of silently scoring the test split."""
    with pytest.raises(ValueError, match="validation-fraction"):
        main(["--output", str(tmp_path), flag])
