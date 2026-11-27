"""Regression for selecting the best evaluated ETTh1 Anderson iterate."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

from kaiwu.torch_plugin import QDEQConfig


@pytest.fixture(scope="module")
def anderson_deq():
    example = (
        Path(__file__).resolve().parents[1]
        / "example" / "qdeq" / "simple" / "etth1_common.py"
    )
    spec = spec_from_file_location("etth1_anderson_seed_example", example)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AndersonDEQ


@pytest.mark.parametrize("stop_mode", ["abs", "rel"])
def test_anderson_keeps_seed_when_acceleration_increases_residual(anderson_deq, stop_mode):
    config = QDEQConfig(f_thres=3, stop_mode=stop_mode, anderson_lam=1e-4)
    solver = anderson_deq(torch.nn.Identity(), config)
    initial = torch.zeros((1, 1), dtype=torch.float64)
    evaluations = 0

    def fixed_point_map(guess):
        nonlocal evaluations
        evaluations += 1
        return 0.9 * torch.tanh(guess) + 0.5

    result, stats = solver.solve_fixed_point(fixed_point_map, initial)
    assert evaluations == 3

    # The second seed is closer than the first. The only accelerated update
    # overshoots both, even though the fixed-point map is contractive.
    better_seed = fixed_point_map(initial)
    seed_image = fixed_point_map(better_seed)
    expected_residual = (seed_image - better_seed).norm().item()
    if stop_mode == "rel":
        expected_residual /= seed_image.norm().item() + 1e-9

    torch.testing.assert_close(result, better_seed)
    assert stats["residual"] == pytest.approx(expected_residual)
    assert stats["nstep"] == 1
