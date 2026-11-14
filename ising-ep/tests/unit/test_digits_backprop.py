"""Tests for the backpropagation MLP baseline on the Digits split.

The baseline needs no solver, so these tests train on the real data; three
epochs take well under a second.
"""

# Test functions document intent in their summary.
# pylint: disable=missing-param-doc,missing-return-doc

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from examples.digits_kaiwu.backprop import (
    BackpropConfig,
    aggregate,
    build_mlp,
    build_parser,
    config_from_args,
    predict,
    run_seed,
)
from examples.digits_kaiwu.data import DigitsDataset, load_digits_splits


def _datasets() -> tuple[DigitsDataset, DigitsDataset]:
    """Return the default train and test datasets."""
    splits = load_digits_splits()
    return DigitsDataset(splits.train), DigitsDataset(splits.test)


def test_mlp_mirrors_the_ep_hidden_widths() -> None:
    """``64 -> hidden -> 10`` with a tanh after every hidden layer."""
    model = build_mlp((32, 16), seed=0)

    linears = [layer for layer in model if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (64, 32),
        (32, 16),
        (16, 10),
    ]
    assert sum(isinstance(layer, nn.Tanh) for layer in model) == 2


def test_initialization_is_seeded_and_leaves_the_global_generator_alone() -> None:
    """The same seed gives the same weights, and torch's RNG is restored."""
    before = torch.random.get_rng_state()

    first = build_mlp((8,), seed=4).state_dict()
    second = build_mlp((8,), seed=4).state_dict()
    other = build_mlp((8,), seed=5).state_dict()

    assert all(torch.equal(first[key], second[key]) for key in first)
    assert not torch.equal(first["0.weight"], other["0.weight"])
    assert torch.equal(torch.random.get_rng_state(), before)


def test_training_lowers_the_loss_and_beats_chance(tmp_path: Path) -> None:
    """Three epochs are enough to be far above the 10% chance level."""
    train, test = _datasets()

    status = run_seed(BackpropConfig(epochs=3), train, test, 0, tmp_path)

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))["epochs"]
    losses = [row["train_loss"] for row in history]
    assert losses == sorted(losses, reverse=True)
    assert status["test_accuracy"] > 0.5
    assert all(row["epoch_seconds"] > 0 for row in history)


def test_run_seed_writes_history_model_and_metrics(tmp_path: Path) -> None:
    """Each seed leaves its curve, its weights and its test metrics."""
    train, test = _datasets()

    run_seed(BackpropConfig(epochs=1), train, test, 0, tmp_path)

    for relative in ("history.json", "model.pt", "status.json", "test/metrics.json"):
        assert (tmp_path / relative).is_file(), relative


def test_same_seed_reproduces_the_accuracy_curve(tmp_path: Path) -> None:
    """Initialization and shuffling both derive from the seed."""
    train, test = _datasets()

    first = run_seed(BackpropConfig(epochs=2), train, test, 1, tmp_path / "a")
    second = run_seed(BackpropConfig(epochs=2), train, test, 1, tmp_path / "b")

    assert first["test_accuracy"] == second["test_accuracy"]
    assert first["final_train_loss"] == second["final_train_loss"]


def test_saved_state_dict_reloads_to_identical_predictions(tmp_path: Path) -> None:
    """``model.pt`` restores a network that predicts exactly as the trained one."""
    train, test = _datasets()
    config = BackpropConfig(epochs=1)
    run_seed(config, train, test, 0, tmp_path)

    restored = build_mlp(config.hidden_sizes, seed=123)
    restored.load_state_dict(torch.load(tmp_path / "model.pt"))
    metrics = json.loads((tmp_path / "test" / "metrics.json").read_text(encoding="utf-8"))

    assert float((predict(restored, test.images) == test.labels).double().mean()) == (
        metrics["accuracy"]
    )


def test_aggregate_reports_cross_seed_statistics() -> None:
    """Mean and spread of train/test accuracy and epoch time."""
    rows = [
        {"seed": 0, "train_accuracy": 0.9, "test_accuracy": 0.8, "mean_epoch_seconds": 1.0},
        {"seed": 1, "train_accuracy": 1.0, "test_accuracy": 0.9, "mean_epoch_seconds": 3.0},
    ]

    summary = aggregate(rows)

    assert summary["seeds"] == [0, 1]
    assert summary["test_accuracy"]["mean"] == pytest.approx(0.85)
    assert summary["mean_epoch_seconds"]["mean"] == pytest.approx(2.0)


def test_cli_defaults_match_the_ep_training_budget() -> None:
    """Same widths, epochs, batch size and split as the EP Digits run."""
    config = config_from_args(build_parser().parse_args(["--output", "x"]))

    assert config.hidden_sizes == (64,)
    assert (config.epochs, config.batch_size) == (5, 10)
    assert (config.split_seed, config.test_fraction) == (0, 0.2)
    assert config.seeds == (0, 1, 2)
