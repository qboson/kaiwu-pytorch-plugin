"""The forecast CLI seed controls QDEQ's random search directions."""

import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

from kaiwu.torch_plugin import QDEQConfig
from kaiwu.torch_plugin import qdeq


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"


class ExactSampler:
    """Solve the tiny test Ising system without a license or global RNG."""

    def solve(self, ising_mat):
        """Return the spin row maximizing the quadratic objective."""
        rows = np.array(list(itertools.product((-1.0, 1.0), repeat=ising_mat.shape[0])))
        energies = np.einsum("bi,ij,bj->b", rows, ising_mat, rows)
        return rows[np.argmax(energies)]


def load_module(name, path):
    """Load an example script without changing the Python import path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forecast_seed_reproduces_solver_outputs(monkeypatch, tmp_path, capsys):
    """Repeated CLI calls ignore the caller's previous PyTorch RNG state."""
    common = load_module("forecast_seed_common", EXAMPLE_DIR / "etth1_common.py")
    monkeypatch.setitem(sys.modules, "etth1_common", common)
    script = load_module("forecast_seed_cli", EXAMPLE_DIR / "simple_forecast_example.py")
    config = QDEQConfig(f_thres=2, num_basis=2, coeff_bits=2, fd_eps=1e-2)

    def model_factory(*_args):
        return common.DEQForecaster(
            4, 2, ExactSampler(), config, d_model=4, n_heads=1, d_ff=8
        )

    with torch.random.fork_rng():
        torch.manual_seed(123)
        checkpoint = tmp_path / "model.pt"
        torch.save({"state_dict": model_factory().state_dict()}, checkpoint)
    history = torch.tensor([[[1.0, 4.0], [2.0, 1.0], [4.0, 3.0], [3.0, 2.0]]])
    loader = [(history, torch.zeros(1, 2, 2))]
    args = SimpleNamespace(
        data_path=None, checkpoint=checkpoint, sampler="sa", seed=17
    )
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(script, "build_loaders", lambda _path: {"test": loader})
    monkeypatch.setattr(script, "build_sampler", lambda *_args: ExactSampler())
    monkeypatch.setattr(script, "DEQForecaster", model_factory)
    forecasts = []
    directions = []
    search_directions = qdeq._search_directions

    def record_directions(*args):
        result = search_directions(*args)
        directions.append(result.clone())
        return result

    monkeypatch.setattr(qdeq, "_search_directions", record_directions)

    def evaluate(model, batches):
        model.eval()
        with torch.no_grad():
            forecasts.append(model(batches[0][0]).clone())
        return {"mse": 0.0, "mae": 0.0, "steps": 2.0}

    monkeypatch.setattr(script, "evaluate", evaluate)
    runs = []
    for previous_seed in (101, 202):
        torch.manual_seed(previous_seed)
        directions.clear()
        script.main()
        runs.append(list(directions))
        capsys.readouterr()
    assert runs[0] and len(runs[0]) == len(runs[1])
    for first, second in zip(*runs):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(forecasts[0], forecasts[1], rtol=0, atol=0)
    args.seed = 18
    directions.clear()
    script.main()
    capsys.readouterr()
    assert any(not torch.equal(first, second) for first, second in zip(runs[0], directions))
