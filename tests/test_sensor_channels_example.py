"""Check that training recovers the simulated channels on independent samples."""

from pathlib import Path
from runpy import run_path

import pytest


@pytest.mark.parametrize(("train_seed", "test_seed"), [(42, 43), (17, 23)])
def test_learned_channels_keep_the_held_out_signal(
    train_seed: int, test_seed: int
) -> None:
    """Check channel recovery and compare held-out error with both baselines."""
    root = Path(__file__).resolve().parents[1]
    example = root / "example/feature_selection/sensor_channels.py"
    run_example = run_path(str(example))["run_example"]
    selected, full_mse, selected_mse, baseline_mse = run_example(train_seed, test_seed)

    assert selected == [1, 4]
    assert selected_mse < 0.05 * baseline_mse
    assert abs(selected_mse - full_mse) < 0.01 * baseline_mse
