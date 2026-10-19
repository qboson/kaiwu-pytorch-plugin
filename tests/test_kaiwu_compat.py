"""Exercise the installed Kaiwu SDK and the plugin's sampler examples."""

import importlib
import importlib.util
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

import kaiwu as kw
from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine
from kaiwu.torch_plugin.maifs import qubo


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cim_without_remote_access(monkeypatch, tmp_path):
    """Use the real CIM constructor without credentials or remote requests."""
    cim_module = importlib.import_module("kaiwu.cim._cim_optimizer")
    monkeypatch.setattr(cim_module, "ensure_license", lambda: None)
    monkeypatch.setattr(
        kw.cim.CIMOptimizer, "_get_authorization", staticmethod(lambda: "test")
    )
    monkeypatch.setattr(kw.common.CheckpointManager, "save_dir", str(tmp_path))


@pytest.mark.usefixtures("cim_without_remote_access")
def test_feature_selection_cim_default_mode(monkeypatch, tmp_path):
    """The default feature-selection configuration passes SDK validation."""
    def solve_locally(sampler, matrix):
        assert sampler.task_mode == kw.cim.TaskMode.OPTIMIZATION
        return kw.classical.BruteForceOptimizer().solve(matrix)

    monkeypatch.setattr(kw.cim.CIMOptimizer, "_solve", solve_locally)
    result = qubo._solve_ising_kaiwu_cim(  # pylint: disable=protected-access
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        target_precision=8,
        max_precision=8,
        save_dir=tmp_path,
        cleanup_records=False,
    )
    np.testing.assert_array_equal(result, [[1, 1]])


@pytest.mark.parametrize("use_precision_reducer", [False, True])
@pytest.mark.usefixtures("cim_without_remote_access")
def test_dplm_cim_sampler(use_precision_reducer):
    """The DPLM example constructs raw and precision-reduced CIM samplers."""
    path = PROJECT_ROOT / "example/qdiffusion/dplm/models/sampler.py"
    spec = importlib.util.spec_from_file_location("dplm_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sampler = module.build_bm_sampler(
        sampler_type="cim",
        sampler_kwargs={
            "task_mode": "SAMPLING",
            "sample_number": 10,
            "use_precision_reducer": use_precision_reducer,
        },
    )
    expected_type = (
        kw.preprocess.PrecisionReducer if use_precision_reducer else kw.cim.CIMOptimizer
    )
    assert isinstance(sampler, expected_type)


def test_rbm_example_imports():
    """The basic RBM example imports with the installed SDK."""
    runpy.run_path(str(PROJECT_ROOT / "example/run_rbm.py"))


@pytest.mark.parametrize("model_type", [BoltzmannMachine, RestrictedBoltzmannMachine])
@pytest.mark.parametrize("reduce_precision", [False, True])
def test_sdk_sampling_and_training(model_type, reduce_precision):
    """Run SDK solving, sample conversion, and a model update end to end."""
    if model_type is RestrictedBoltzmannMachine:
        model = model_type(2, 2, device="cpu")
    else:
        model = model_type(4, device="cpu")
    with torch.no_grad():
        model.quadratic_coef.fill_(0.25)
        model.linear_bias.copy_(torch.tensor([0.25, -0.5, 0.75, -1.0]))
    sampler = kw.classical.BruteForceOptimizer()
    if reduce_precision:
        sampler = kw.preprocess.PrecisionReducer(
            sampler, precision=8, truncated_precision=8, target_bits=100
        )

    samples = model.sample(sampler)
    assert samples.ndim == 2 and samples.shape[1] == 4
    assert torch.all((samples == 0) | (samples == 1))
    states = torch.tensor(list(np.ndindex(2, 2, 2, 2)), dtype=torch.float32)
    torch.testing.assert_close(model(samples), model(states).min().expand(len(samples)))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    previous_bias = model.linear_bias.detach().clone()
    loss = model.objective(1 - samples, samples)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(previous_bias, model.linear_bias)


def test_sdk_conditional_sampling():
    """Conditional BM sampling preserves each supplied visible state."""
    model = BoltzmannMachine(4, device="cpu")
    visible = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    sampler = kw.classical.BruteForceOptimizer()
    for state in visible:
        samples = model.condition_sample(sampler, state.unsqueeze(0))
        torch.testing.assert_close(samples[:, :2], state.expand(len(samples), -1))


@pytest.mark.parametrize("model_type", [BoltzmannMachine, RestrictedBoltzmannMachine])
def test_model_to_preserves_pytorch_arguments(model_type):
    """Models accept dtype-only and tensor-reference moves like PyTorch modules."""
    if model_type is RestrictedBoltzmannMachine:
        model = model_type(2, 2, device="cpu")
    else:
        model = model_type(4, device="cpu")
    assert model.to(dtype=torch.float64) is model
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    assert model.device == torch.device("cpu")
    assert model.to(torch.zeros(1, dtype=torch.float32)) is model
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert model.device == torch.device("cpu")


@pytest.mark.parametrize("via_parent", [False, True])
def test_gbrbm_sampling_after_dtype_conversion(via_parent):
    """Sampling follows direct and parent-initiated dtype changes."""
    model = GaussianBernoulliRestrictedBoltzmannMachine(2, 2, device="cpu")
    target = model
    if via_parent:
        target = torch.nn.Module()
        target.model = model
    for dtype in (torch.float64, torch.float32):
        target.to("cpu", dtype=dtype)
        samples = model.gibbs_sample(n_step=1, n_sample=2)
        assert samples.shape == (2, 4)
        assert samples.dtype == dtype
        assert torch.isfinite(model(samples)).all()
