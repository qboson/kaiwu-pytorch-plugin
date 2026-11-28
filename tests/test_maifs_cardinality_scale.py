"""Scale-equivariance and CIM quantization regressions for MAIFS cardinality selection."""

from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import kaiwu as kw
from kaiwu.torch_plugin.maifs import qubo
from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper


def _sdk_minimum(matrix: np.ndarray) -> np.ndarray:
    """Return the exact ground-state spin row under ``kw.common.hamiltonian``."""
    spins = np.array(list(product((-1, 1), repeat=len(matrix))))
    energies = kw.common.hamiltonian(matrix, spins)
    return spins[[int(np.argmin(energies))]]


@pytest.fixture(name="exact_offline_solvers")
def _exact_offline_solvers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace licensed SA/CIM annealing calls with exact SDK energy enumeration."""
    sa_cls = kw.classical.SimulatedAnnealingOptimizer

    def fake_single_process(
        instance,
        ising_matrix=None,
        init_solution=None,
        rand_seed=None,
    ):
        del init_solution, rand_seed
        instance.set_matrix(ising_matrix)
        return _sdk_minimum(instance.matrix)

    monkeypatch.setattr(sa_cls, "single_process_solve", fake_single_process)
    monkeypatch.setattr(
        kw.cim,
        "CIMOptimizer",
        lambda **_kwargs: SimpleNamespace(solve=_sdk_minimum),
    )
    monkeypatch.setattr(kw.common.CheckpointManager, "save_dir", None)


@pytest.mark.parametrize("solver", ["local_search", "sa"])
@pytest.mark.parametrize("scale", [1.0, 10.0, 1e4, 1e-4, 1e-8])
def test_cardinality_selection_is_equivariant_across_target_scales(
    exact_offline_solvers: None,
    solver: str,
    scale: float,
) -> None:
    """Positive target rescaling y -> c * y must preserve exact cardinality_k selection."""
    del exact_offline_solvers
    x_data = torch.eye(3, dtype=torch.float64)
    y_base = torch.tensor([[2.0], [1.5], [0.0]], dtype=torch.float64)
    model = nn.Linear(3, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(scale * torch.tensor([[2.0, 1.5, 0.0]], dtype=torch.float64))

    selector = FeatureSelectionWrapper(
        model,
        feature_dim=3,
        cardinality_k=1,
        solver=solver,
    )
    selected = selector.update_mask([(x_data, scale * y_base)], nn.MSELoss())

    assert selected.tolist() == [1, 0, 0]
    assert selector.num_selected() == 1


@pytest.mark.parametrize("scale", [1.0, 10.0, 1e-3, 1e-4])
def test_kaiwu_cim_cardinality_selection_preserves_small_scale_feature_ranking(
    exact_offline_solvers: None,
    tmp_path,
    scale: float,
) -> None:
    """CIM quantization must not drown small-scale feature derivatives in fixed penalty."""
    del exact_offline_solvers
    x_data = torch.eye(3, dtype=torch.float64)
    y_base = torch.tensor([[2.0], [1.5], [0.0]], dtype=torch.float64)
    model = nn.Linear(3, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(scale * torch.tensor([[2.0, 1.5, 0.0]], dtype=torch.float64))

    selector = FeatureSelectionWrapper(
        model,
        feature_dim=3,
        cardinality_k=1,
        solver="kaiwu_cim",
        solver_kwargs={
            "save_dir": tmp_path,
            "cleanup_records": True,
            "target_precision": 8,
            "max_precision": 8,
            "max_bits": 64,
        },
    )
    selected = selector.update_mask([(x_data, scale * y_base)], nn.MSELoss())

    assert selected.tolist() == [1, 0, 0]
    assert selector.num_selected() == 1


@pytest.mark.parametrize("hessian_mode", ["full", "diagonal"])
@pytest.mark.parametrize("scale", [1.0, 50.0, 1e-8])
def test_multi_feature_cardinality_scale_equivariance_across_hessian_modes(
    exact_offline_solvers: None,
    hessian_mode: str,
    scale: float,
) -> None:
    """Selecting k=2 interacting features remains scale-equivariant in full and diagonal modes."""
    del exact_offline_solvers
    x_data = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.1, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    weights = scale * torch.tensor([[3.0, 2.2, 1.1, 0.0]], dtype=torch.float64)
    y_data = x_data @ weights.t()
    model = nn.Linear(4, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(weights)

    selector = FeatureSelectionWrapper(
        model,
        feature_dim=4,
        cardinality_k=2,
        solver="sa",
    )
    selected = selector.update_mask(
        [(x_data, y_data)],
        nn.MSELoss(),
        hessian_mode=hessian_mode,
    )

    assert selected.tolist() == [1, 1, 0, 0]
    assert selector.num_selected() == 2


def test_flat_zero_loss_cardinality_penalty_falls_back_cleanly() -> None:
    """Exact-zero gradient, Hessian, and lambda_reg still enforce cardinality_k."""
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    selector = FeatureSelectionWrapper(
        model,
        feature_dim=4,
        lambda_reg=0.0,
        cardinality_k=2,
        solver="local_search",
    )
    selected = selector.update_mask(
        [(torch.zeros(4, 4), torch.zeros(4, 1))],
        nn.MSELoss(),
    )

    assert int(selected.sum()) == 2
    assert selector.num_selected() == 2
    ising = qubo.QuadraticLinearSolver().solve(
        *selector._build_qubo(np.zeros(4), np.zeros((4, 4)))  # pylint: disable=protected-access
    )
    assert np.isfinite(ising).all()
