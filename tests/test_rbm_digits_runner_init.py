"""Contract test for the ``RBMRunner.__init__`` sampler and device setup."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from kaiwu.classical import SimulatedAnnealingOptimizer

REPO = Path(__file__).resolve().parents[1]
RBM_DIGITS = REPO / "example" / "rbm_digits" / "rbm_digits.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("rbm_digits_runner_uut", RBM_DIGITS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_defaults_select_the_sa_sampler_with_the_documented_knobs(capsys):
    runner_module = _load_runner()

    runner = runner_module.RBMRunner(
        n_components=16,
        learning_rate=0.05,
        batch_size=8,
        n_iter=2,
        verbose=True,
        plot_img=True,
        random_state=11,
    )

    assert runner.n_components == 16
    assert runner.learning_rate == 0.05
    assert runner.batch_size == 8
    assert runner.n_iter == 2
    assert runner.verbose is True
    assert runner.plot_img is True
    assert runner.random_state == 11
    assert runner.rbm is None
    assert runner.device == torch.device("cpu")
    assert isinstance(runner.sampler, SimulatedAnnealingOptimizer)
    assert runner.sampler._alpha == 0.999
    assert runner.sampler._size_limit == 100
    assert "Using Simulated Annealing optimizer" in capsys.readouterr().out


def test_sklearn_parameter_introspection_sees_the_configuration():
    runner_module = _load_runner()
    runner = runner_module.RBMRunner(n_components=8, use_cim=False)

    params = runner.get_params()

    assert params["n_components"] == 8
    assert params["use_cim"] is False
    assert params["learning_rate"] == 0.1
    assert params["n_iter"] == 30
