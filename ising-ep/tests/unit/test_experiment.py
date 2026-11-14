"""Tests for run orchestration: seed derivation, artifacts and aggregation.

A fake solver keeps these fast and deterministic. The real solver is exercised
by ``tests/integration/test_kaiwu_real_sdk.py``.
"""

# Test functions document intent in their summary; pytest fixtures and
# parametrised values do not benefit from Args sections. CyclingSolver mirrors
# the KaiwuSolver signature, so some arguments are deliberately unused.
# pylint: disable=missing-param-doc,missing-return-doc,unused-argument

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.mnist_kaiwu.experiment import (
    RunConfig,
    aggregate,
    build_optimizer,
    build_parser,
    config_from_args,
    derived_int,
    run_seed,
)
from examples.mnist_kaiwu.train import train_epoch
from ising_ep.kaiwu_solver import AnnealSettings
from ising_ep.model import IsingEPModel

from tests.conftest import PRIMARY_ANNEAL, FakeSolver, to_numpy

N_HIDDEN = 3
N_CLASSES = 2
SPINS_PER_CLASS = 2
N_PHYSICAL = N_HIDDEN + N_CLASSES * SPINS_PER_CLASS


class CyclingSolver:
    """Returns alternating candidate states so the two phases always differ."""

    def __init__(self) -> None:
        self.calls = 0

    def solve(
        self,
        Q: Any,
        *,
        seed: int,
        init_solution: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a state that depends on the seed and the call parity."""
        self.calls += 1
        rng = np.random.default_rng(int(seed) + 7919 * (self.calls % 2))
        state = rng.choice([-1.0, 1.0], size=N_PHYSICAL)
        return np.concatenate([state, [1.0]]).reshape(1, -1)


def _config(**overrides: Any) -> RunConfig:
    """Return a tiny run configuration for the fake network."""
    defaults = {
        "seeds": (0,),
        "hidden_sizes": (N_HIDDEN,),
        "epochs": 1,
        "batch_size": 2,
        "lr": 0.05,
        "beta": 0.5,
        "anneal": AnnealSettings(candidate_count=1),
        "n_in": 4,
        "n_classes": N_CLASSES,
        "spins_per_class": SPINS_PER_CLASS,
    }
    defaults.update(overrides)
    return RunConfig(**defaults)


def _dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a 4-sample train split and a 2-sample test split."""
    rng = np.random.default_rng(0)
    return (
        rng.random((4, 4)),
        np.array([0, 1, 0, 1], dtype=np.int64),
        rng.random((2, 4)),
        np.array([0, 1], dtype=np.int64),
    )


def _run(tmp_path: Path, config: RunConfig, seed: int = 0) -> dict[str, Any]:
    """Run one seed against the fake solver under ``tmp_path``."""
    train_x, train_y, test_x, test_y = _dataset()
    seed_dir = tmp_path / "seeds" / str(seed)
    return run_seed(
        config, CyclingSolver(), train_x, train_y, test_x, test_y, seed, seed_dir
    )


def test_derived_int_is_stable_and_domain_separated() -> None:
    """Seeds are reproducible, and different streams do not collide."""
    assert derived_int(3, "model") == derived_int(3, "model")
    assert derived_int(3, "model") != derived_int(3, "shuffle")
    assert derived_int(3, "pair", 0, 1) != derived_int(3, "pair", 1, 0)
    assert 0 <= derived_int(3, "pair", 0, 1) < 2**31


def test_run_seed_writes_the_expected_artifact_tree(tmp_path: Path) -> None:
    """Each seed produces its checkpoints, per-epoch dynamics and test output."""
    config = _config(epochs=2)

    _run(tmp_path, config)

    seed_dir = tmp_path / "seeds" / "0"
    for relative in (
        "initial-model.npz",
        "checkpoint.npz",
        "status.json",
        "epochs/0/model.npz",
        "epochs/0/dynamics.json",
        "epochs/1/dynamics.json",
        "test/metrics.json",
        "test/predictions.json",
    ):
        assert (seed_dir / relative).is_file(), relative


def test_status_reports_accuracy_learning_evidence_and_call_counts(tmp_path: Path) -> None:
    """The status file carries what a reader needs to judge the run."""
    config = _config(epochs=2)

    status = _run(tmp_path, config)

    assert status["seed"] == 0
    assert 0.0 <= status["accuracy"] <= 1.0
    assert len(status["per_class"]) == N_CLASSES
    assert set(status["learning_evidence"]) >= {"weight_changes", "passed"}
    assert status["solver_calls"] == {
        # 4 training samples x 2 epochs, then one call per test image.
        "train_free": 8,
        "train_nudged": 8,
        "test_evaluation": 2,
    }
    on_disk = json.loads((tmp_path / "seeds" / "0" / "status.json").read_text(encoding="utf-8"))
    assert on_disk["accuracy"] == status["accuracy"]


def test_training_moves_the_parameters(tmp_path: Path) -> None:
    """A run that learns nothing would be a silent failure, so assert movement."""
    status = _run(tmp_path, _config())

    assert status["learning_evidence"]["weight_changes"][0] > 0.0
    assert status["learning_evidence"]["weight_changes"][1] > 0.0
    with np.load(tmp_path / "seeds" / "0" / "initial-model.npz") as initial:
        with np.load(tmp_path / "seeds" / "0" / "checkpoint.npz") as final:
            assert not np.array_equal(initial["weights.0"], final["weights.0"])


def test_same_seed_reproduces_identical_artifacts(tmp_path: Path) -> None:
    """Two runs of one seed agree bit for bit."""
    config = _config()

    first = _run(tmp_path / "a", config)
    second = _run(tmp_path / "b", config)

    assert first == second
    left = (tmp_path / "a" / "seeds" / "0" / "status.json").read_bytes()
    right = (tmp_path / "b" / "seeds" / "0" / "status.json").read_bytes()
    assert left == right


def test_different_seeds_diverge(tmp_path: Path) -> None:
    """Seeds must not collapse onto the same initialization."""
    config = _config()

    _run(tmp_path, config, seed=0)
    _run(tmp_path, config, seed=1)

    with np.load(tmp_path / "seeds" / "0" / "initial-model.npz") as first:
        with np.load(tmp_path / "seeds" / "1" / "initial-model.npz") as second:
            assert not np.array_equal(first["weights.0"], second["weights.0"])


def test_aggregate_summarises_accuracy_and_per_class_across_seeds() -> None:
    """Cross-seed statistics use the population standard deviation."""
    results = [
        {
            "seed": index,
            "accuracy": accuracy,
            "tie_rate": 0.1,
            "per_class": [
                {"class": 0, "accuracy": accuracy},
                {"class": 1, "accuracy": 1.0 - accuracy},
            ],
            "learning_evidence": {"passed": True},
        }
        for index, accuracy in enumerate((0.58, 0.70, 0.59))
    ]

    summary = aggregate(results, n_classes=2)

    assert summary["n_seeds"] == 3
    assert summary["seeds"] == [0, 1, 2]
    assert summary["accuracy"]["mean"] == pytest.approx(0.6233333333333333)
    assert summary["accuracy"]["min"] == pytest.approx(0.58)
    assert summary["accuracy"]["max"] == pytest.approx(0.70)
    assert summary["accuracy"]["std"] == pytest.approx(0.05436502143433363)
    assert summary["per_class"][0]["mean"] == pytest.approx(0.6233333333333333)
    assert summary["learning_evidence_passed"] is True


def test_aggregate_accepts_a_single_seed() -> None:
    """Smoke runs use one seed, which must not be rejected."""
    summary = aggregate(
        [
            {
                "seed": 0,
                "accuracy": 0.5,
                "tie_rate": 0.0,
                "per_class": [{"class": 0, "accuracy": 0.5}],
                "learning_evidence": {"passed": False},
            }
        ],
        n_classes=1,
    )

    assert summary["n_seeds"] == 1
    assert summary["accuracy"]["std"] == pytest.approx(0.0)
    assert summary["learning_evidence_passed"] is False


def test_cli_defaults_match_the_documented_primary_run() -> None:
    """The default configuration is the 784-64-40, five-epoch, three-seed run."""
    args = build_parser().parse_args(["--output", "runs/x"])

    config = config_from_args(args)

    assert config.seeds == (0, 1, 2)
    assert (config.hidden_sizes, config.epochs, config.batch_size) == ((64,), 5, 10)
    assert (config.lr, config.beta) == (0.01, 1.0)
    assert config.anneal == PRIMARY_ANNEAL
    assert config.n_classes * config.spins_per_class == 40


def test_cli_parses_a_seed_list() -> None:
    """Seeds arrive as a comma-separated list."""
    args = build_parser().parse_args(["--output", "runs/x", "--seeds", "4, 5"])

    assert config_from_args(args).seeds == (4, 5)


def test_config_json_records_model_training_and_anneal_settings() -> None:
    """``config.json`` is the reproduction record for a run."""
    payload = _config().to_json()

    assert payload["model"]["n_output"] == N_CLASSES * SPINS_PER_CLASS
    assert payload["model"]["init_gains"] == [1.0, 1.0]
    assert payload["training"]["beta"] == 0.5
    assert payload["training"]["lr_weights"] == [0.05, 0.05]
    assert payload["training"]["lr_biases"] == [0.05, 0.05]
    assert payload["training"]["weight_decay"] == 0.0
    assert payload["anneal"]["candidate_count"] == 1
    assert payload["created_at_utc"].endswith("Z")


def test_build_optimizer_defaults_to_one_rate_per_parameter() -> None:
    """Unspecified per-layer lists repeat the global learning rate."""
    model = IsingEPModel(n_in=4, hidden_sizes=3, n_classes=2, spins_per_class=2, seed=0)
    optimizer = build_optimizer(model, _config())

    assert len(optimizer.param_groups) == 4
    assert [group["lr"] for group in optimizer.param_groups] == [0.05, 0.05, 0.05, 0.05]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0, 0.0, 0.0, 0.0]


def test_zero_output_learning_rate_leaves_that_weight_unchanged() -> None:
    """A zero rate on the last connection freezes that matrix for one step."""
    model = IsingEPModel(n_in=3, hidden_sizes=2, n_classes=2, spins_per_class=2, seed=5)
    before = model.parameter_snapshot()
    config = _config(lr_weights=(0.1, 0.0), lr_biases=(0.0, 0.0))
    optimizer = build_optimizer(model, config)
    free = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    nudged = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0])

    train_epoch(
        model,
        optimizer,
        FakeSolver(free, nudged),
        np.ones((1, 3)),
        np.zeros(1, dtype=np.int64),
        order=[0],
        beta=0.5,
        batch_size=1,
        pair_seed_fn=lambda _index: 1,
    )

    assert not np.array_equal(to_numpy(model.weights[0]), to_numpy(before["weights.0"]))
    np.testing.assert_allclose(to_numpy(model.weights[1]), to_numpy(before["weights.1"]))


def test_per_layer_rate_lists_must_match_the_connection_count() -> None:
    """A list that does not cover every connection is rejected."""
    model = IsingEPModel(n_in=4, hidden_sizes=3, n_classes=2, spins_per_class=2, seed=0)

    with pytest.raises(ValueError, match="lr_weights must have length 2"):
        build_optimizer(model, _config(lr_weights=(0.1,)))


def test_cli_parses_per_layer_rates_and_init_gains() -> None:
    """Comma-separated floats become one entry per connection."""
    args = build_parser().parse_args(
        [
            "--output",
            "runs/x",
            "--lr-weights",
            "0.1,0.05",
            "--init-gains",
            "1,0.5",
            "--weight-decay",
            "1e-6",
        ]
    )
    config = config_from_args(args)

    assert config.lr_weights == (0.1, 0.05)
    assert config.init_gains == (1.0, 0.5)
    assert config.weight_decay == pytest.approx(1e-6)
    assert config.resolved_lr_biases() == (0.01, 0.01)


def _validation_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a 2-sample validation split for the fake network."""
    rng = np.random.default_rng(1)
    return rng.random((2, 4)), np.array([0, 1], dtype=np.int64)


def test_default_config_json_has_no_validation_block() -> None:
    """Default runs keep their original config layout."""
    assert "validation" not in _config().to_json()
    assert _config(skip_test=True).to_json()["validation"]["skip_test"] is True


def test_eval_every_epoch_writes_validation_json(tmp_path: Path) -> None:
    """Each epoch gets a validation record when requested."""
    train_x, train_y, test_x, test_y = _dataset()
    val_x, val_y = _validation_data()
    config = _config(epochs=2, validation_per_class=1, eval_every_epoch=True)
    seed_dir = tmp_path / "seeds" / "0"

    status = run_seed(
        config, CyclingSolver(), train_x, train_y, test_x, test_y, 0, seed_dir,
        validation_images=val_x, validation_labels=val_y,
    )

    for epoch in (0, 1):
        record = json.loads(
            (seed_dir / "epochs" / str(epoch) / "validation.json").read_text(encoding="utf-8")
        )
        assert "accuracy" in record and "tie_rate" in record
    assert (seed_dir / "test" / "metrics.json").is_file()
    assert status["solver_calls"]["validation_evaluation"] == 4


def test_skip_test_never_writes_test_artifacts(tmp_path: Path) -> None:
    """Tuning runs report validation accuracy and leave ``test/`` absent."""
    train_x, train_y, test_x, test_y = _dataset()
    val_x, val_y = _validation_data()
    config = _config(validation_per_class=1, skip_test=True)
    seed_dir = tmp_path / "seeds" / "0"

    status = run_seed(
        config, CyclingSolver(), train_x, train_y, test_x, test_y, 0, seed_dir,
        validation_images=val_x, validation_labels=val_y,
    )

    assert not (seed_dir / "test").exists()
    assert status["evaluated_on"] == "validation"
    assert status["solver_calls"]["test_evaluation"] == 0
    assert (seed_dir / "epochs" / "0" / "validation.json").is_file()


def test_validation_requires_a_validation_split(tmp_path: Path) -> None:
    """Asking to validate without data is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="validation split"):
        _run(tmp_path, _config(skip_test=True))


def test_cli_parses_validation_flags() -> None:
    """The validation switches reach the run configuration."""
    args = build_parser().parse_args(
        ["--output", "runs/x", "--validation-per-class", "10", "--eval-every-epoch", "--skip-test"]
    )
    config = config_from_args(args)

    assert (config.validation_per_class, config.eval_every_epoch, config.skip_test) == (
        10,
        True,
        True,
    )
