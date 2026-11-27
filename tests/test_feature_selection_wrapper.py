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


def test_update_mask_with_finite_fp16_hessian() -> None:
    """测试半精度 Hessian 对称化不会使掩码更新失败。"""
    selector = FeatureSelectionWrapper(nn.Identity(), feature_dim=2)
    features = torch.tensor([[192.0, 1.0]], dtype=torch.float16)
    targets = torch.tensor([[192.0, 0.0]], dtype=torch.float32)

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction.float() - target).square().mean()

    mask = selector.update_mask([(features, targets)], loss_fn)

    np.testing.assert_array_equal(mask, [1, 0])


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
