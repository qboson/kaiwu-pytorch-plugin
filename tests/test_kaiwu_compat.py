"""Exercise the installed Kaiwu SDK and the plugin's sampler examples."""

import importlib
import importlib.util
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import kaiwu as kw
from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine
from kaiwu.torch_plugin.maifs import qubo
from kaiwu.torch_plugin._kaiwu_compat import PrecisionReducer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cim_without_remote_access(monkeypatch, tmp_path):
    """Use the real CIM constructor without credentials or remote requests."""
    cim_module = importlib.import_module(kw.cim.CIMOptimizer.__module__)
    monkeypatch.setattr(cim_module, "ensure_license", lambda: None)
    monkeypatch.setattr(
        kw.cim.CIMOptimizer, "_get_authorization", staticmethod(lambda: "test")
    )
    monkeypatch.setattr(kw.common.CheckpointManager, "save_dir", str(tmp_path))


@pytest.mark.parametrize("mode_kwargs", [{}, {"task_mode": None}])
@pytest.mark.usefixtures("cim_without_remote_access")
def test_feature_selection_cim_default_mode(monkeypatch, tmp_path, mode_kwargs):
    """The default feature-selection configuration passes SDK validation."""
    def solve_locally(sampler, matrix):
        assert sampler.task_mode == (
            "optimization" if hasattr(kw.cim, "TaskMode") else "quota"
        )
        return kw.classical.BruteForceOptimizer().solve(matrix)

    # 1.3.1 has no separate _solve hook; replace its remote solve boundary.
    solve_method = "_solve" if hasattr(kw.cim.CIMOptimizer, "_solve") else "solve"
    monkeypatch.setattr(kw.cim.CIMOptimizer, solve_method, solve_locally)
    result = qubo._solve_ising_kaiwu_cim(  # pylint: disable=protected-access
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        target_precision=8,
        max_precision=8,
        save_dir=tmp_path,
        cleanup_records=False,
        **mode_kwargs,
    )
    np.testing.assert_array_equal(result, [[1, 1]])


@pytest.mark.parametrize("use_precision_reducer", [False, True])
@pytest.mark.parametrize(
    "task_mode, modern_mode, legacy_mode",
    [
        ("optimization", "optimization", "quota"),
        ("quota", "optimization", "quota"),
        ("SAMPLING", "sampling", "sample"),
        ("sample", "sampling", "sample"),
    ],
)
@pytest.mark.usefixtures("cim_without_remote_access")
def test_dplm_cim_sampler(use_precision_reducer, task_mode, modern_mode, legacy_mode):
    """The DPLM example constructs raw and precision-reduced CIM samplers."""
    path = PROJECT_ROOT / "example/qdiffusion/dplm/models/sampler.py"
    spec = importlib.util.spec_from_file_location("dplm_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sampler = module.build_bm_sampler(
        sampler_type="cim",
        sampler_kwargs={
            "task_mode": task_mode,
            "sample_number": 10,
            "use_precision_reducer": use_precision_reducer,
        },
    )
    expected_type = (
        PrecisionReducer if use_precision_reducer else kw.cim.CIMOptimizer
    )
    assert isinstance(sampler, expected_type)
    if use_precision_reducer:
        sampler = sampler._component  # pylint: disable=protected-access
    expected_mode = modern_mode if hasattr(kw.cim, "TaskMode") else legacy_mode
    assert sampler.task_mode == expected_mode


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
        sampler = PrecisionReducer(
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


@pytest.mark.parametrize(
    "task_mode, modern_mode, legacy_mode",
    [
        ("optimization", "optimization", "quota"),
        ("OPTIMIZATION", "optimization", "quota"),
        ("quota", "optimization", "quota"),
        ("QUOTA", "optimization", "quota"),
        ("sampling", "sampling", "sample"),
        ("SAMPLING", "sampling", "sample"),
        ("sample", "sampling", "sample"),
        ("SAMPLE", "sampling", "sample"),
        (None, None, None),
    ],
)
def test_task_mode_aliases(task_mode, modern_mode, legacy_mode):
    """Old and new configurations select the installed SDK's equivalent mode."""
    compat = importlib.import_module("kaiwu.torch_plugin._kaiwu_compat")
    expected = modern_mode if hasattr(kw.cim, "TaskMode") else legacy_mode
    assert compat.normalize_task_mode(task_mode) == expected


def test_unknown_task_mode_is_rejected():
    """A misspelled mode must not silently submit a different kind of task."""
    compat = importlib.import_module("kaiwu.torch_plugin._kaiwu_compat")
    with pytest.raises(ValueError, match="task mode"):
        compat.normalize_task_mode("samplng")


@pytest.mark.parametrize(
    "task_mode, modern_mode, legacy_mode",
    [
        ("sampling", "sampling", "sample"),
        ("quota", "optimization", "quota"),
        (None, "optimization", "quota"),
    ],
)
@pytest.mark.usefixtures("cim_without_remote_access")
def test_cell_cim_sampler(monkeypatch, tmp_path, task_mode, modern_mode, legacy_mode):
    """The cell example accepts either SDK generation's configuration names."""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "example/qvae_cell"))
    spec = importlib.util.spec_from_file_location(
        "cell_compat_trainer", PROJECT_ROOT / "example/qvae_cell/trainer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        sampler_type="cim", tmp_dir=str(tmp_path), task_name="cell_compat",
        project_no=None, task_mode=task_mode, sample_number=10,
        precision=8, truncated_precision=8, target_bits=100,
    )
    sampler = module.Trainer(args, "cpu").create_sampler()
    assert isinstance(sampler, PrecisionReducer)
    expected_mode = modern_mode if hasattr(kw.cim, "TaskMode") else legacy_mode
    assert sampler._component.task_mode == expected_mode  # pylint: disable=protected-access
