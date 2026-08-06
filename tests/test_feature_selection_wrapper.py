from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from maifs import plugin
from maifs.plugin import FeatureSelectionWrapper


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


def test_fit_weights_updates_mask_with_solver_kwargs(monkeypatch) -> None:
    """测试训练周期到达时会调用 QUBO 求解器并写回新的 mask。"""
    calls: list[dict[str, object]] = []

    def fake_solve_qubo(
        q: np.ndarray,
        c: np.ndarray,
        s0: np.ndarray,
        solver: str,
        **solver_kwargs: object,
    ) -> np.ndarray:
        calls.append(
            {
                "q_shape": q.shape,
                "c_shape": c.shape,
                "s0": s0.copy(),
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
        solver_kwargs={"max_iter": 7, "random_state": 3},
        mask_update_epochs=1,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(selector.model.parameters(), lr=0.01)

    mean_loss = selector.fit_weights(loader, loss_fn, optimizer, epochs=1)

    assert isinstance(mean_loss, float)
    assert len(calls) == 1
    assert calls[0]["q_shape"] == (4, 4)
    assert calls[0]["c_shape"] == (4,)
    assert np.array_equal(calls[0]["s0"], np.ones(4, dtype=int))
    assert calls[0]["solver"] == "sa"
    assert calls[0]["solver_kwargs"] == {"max_iter": 7, "random_state": 3}
    assert np.array_equal(selector.get_support().astype(int), np.array([1, 0, 1, 0]))
    assert selector.selected_indices().tolist() == [0, 2]
    assert selector.num_selected() == 2
