from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

import kaiwu

kaiwu_src_path = os.path.join(src_root, "kaiwu")
if kaiwu_src_path not in kaiwu.__path__:
    kaiwu.__path__ = [kaiwu_src_path] + list(kaiwu.__path__)

for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

from kaiwu.torch_plugin import FeatureSelectionWrapper
from kaiwu.torch_plugin.maifs import plugin


def test_wrapper_applies_mask_before_base_model() -> None:
    """测试特征选择包装器会在基模型前对输入乘以 mask。"""
    selector = FeatureSelectionWrapper(
        nn.Identity(),
        feature_dim=3,
    )
    with torch.no_grad():
        selector.mask.copy_(torch.tensor([1.0, 0.0, 1.0]))

    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    output = selector(x)

    assert torch.equal(
        output,
        torch.tensor(
            [
                [1.0, 0.0, 3.0],
                [4.0, 0.0, 6.0],
            ]
        ),
    )


def test_hard_mask_excludes_nonfinite_values_on_a_feature_axis() -> None:
    """Excluded sensor values must reach the wrapped model as exact zeros."""
    selector = FeatureSelectionWrapper(nn.Identity(), 3, input_feature_axis=1)
    selector.mask.copy_(torch.tensor([1.0, 0.0, 1.0]))
    data = torch.tensor(
        [[[1.0, 2.0, 3.0], [float("nan"), float("inf"), -float("inf")], [4.0, 5.0, 6.0]]],
        requires_grad=True,
    )
    result = selector(data)
    torch.testing.assert_close(
        result, torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]]),
    )
    result.sum().backward()
    torch.testing.assert_close(data.grad, torch.ones_like(data).mul_(selector.mask.view(1, 3, 1)))

    # A nonfinite selected feature is still visible to the wrapped model.
    selected_nonfinite = data.detach().clone()
    selected_nonfinite[0, 2, 0] = float("nan")
    assert torch.isnan(selector(selected_nonfinite)[0, 2, 0])


def test_explicit_soft_mask_retains_gradient_at_zero() -> None:
    """A zero in a differentiable mask can move during derivative evaluation."""
    selector = FeatureSelectionWrapper(nn.Identity(), 3)
    data = torch.tensor([[2.0, 3.0, 4.0]])
    mask = torch.tensor([1.0, 0.0, 0.5], requires_grad=True)
    result = selector.apply_mask(data, mask)
    torch.testing.assert_close(result, torch.tensor([[2.0, 0.0, 2.0]]))
    result.sum().backward()
    torch.testing.assert_close(mask.grad, data[0])


def test_hard_mask_keeps_sparse_inputs_sparse() -> None:
    """Sparse inputs can still be multiplied by the selected feature mask."""
    selector = FeatureSelectionWrapper(nn.Identity(), 2)
    selector.mask.copy_(torch.tensor([0.0, 1.0]))
    result = selector(torch.eye(2).to_sparse())
    assert result.layout == torch.sparse_coo
    torch.testing.assert_close(result.to_dense(), torch.diag(torch.tensor([0.0, 1.0])))


def test_fit_weights_updates_mask_with_solver_kwargs(monkeypatch) -> None:
    """测试训练周期到达时会调用 QUBO 求解器并写回新的 mask。"""
    calls: list[dict[str, object]] = []

    def fake_solve_qubo(
        quadratic_matrix: np.ndarray,
        linear_vector: np.ndarray,
        initial_state: np.ndarray,
        solver: str,
        **solver_kwargs: object,
    ) -> np.ndarray:
        calls.append(
            {
                "quadratic_shape": quadratic_matrix.shape,
                "linear_shape": linear_vector.shape,
                "initial_state": initial_state.copy(),
                "solver": solver,
                "solver_kwargs": dict(solver_kwargs),
            }
        )
        return np.array([1, 0, 1, 0])

    monkeypatch.setattr(plugin, "solve_qubo", fake_solve_qubo)

    x = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 2.0],
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
        ]
    )
    y = torch.tensor([[3.0], [-1.0], [3.0], [-1.0]])
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    selector = FeatureSelectionWrapper(
        nn.Linear(4, 1, bias=False),
        feature_dim=4,
        solver="sa",
        solver_kwargs={"alpha": 0.99, "size_limit": 1, "rand_seed": 3},
        mask_update_epochs=1,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(selector.model.parameters(), lr=0.01)

    mean_loss = selector.fit_weights(loader, loss_fn, optimizer, train_epochs=1)

    assert isinstance(mean_loss, float)
    assert len(calls) == 1
    assert calls[0]["quadratic_shape"] == (4, 4)
    assert calls[0]["linear_shape"] == (4,)
    assert np.array_equal(calls[0]["initial_state"], np.ones(4, dtype=int))
    assert calls[0]["solver"] == "sa"
    assert calls[0]["solver_kwargs"] == {
        "alpha": 0.99,
        "size_limit": 1,
        "rand_seed": 3,
    }
    assert np.array_equal(selector.get_support().astype(int), np.array([1, 0, 1, 0]))
    assert selector.selected_indices().tolist() == [0, 2]
    assert selector.num_selected() == 2
