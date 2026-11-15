from __future__ import annotations

import os
import sys

import numpy as np
import pytest
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


@pytest.mark.parametrize("initial_mode", ["frozen_backbone", "training_child", "eval_model"])
@pytest.mark.parametrize("failure", [None, "loss", "forward"])
def test_mask_derivatives_restore_individual_training_modes(initial_mode, failure) -> None:
    """Derivative evaluation must preserve mixed modes and existing parameter state."""
    model = nn.Sequential(
        nn.BatchNorm1d(3),
        nn.Dropout(p=0.5),
        nn.Linear(3, 1),
    )
    selector = FeatureSelectionWrapper(model, feature_dim=3)
    if initial_mode == "frozen_backbone":
        model[0].eval()
        model[1].eval()
    elif initial_mode == "training_child":
        selector.eval()
        model[1].train()
    else:
        model.eval()

    parameters = list(model.parameters())
    parameters[0].requires_grad_(False)
    for index, parameter in enumerate(parameters):
        if index % 2:
            parameter.grad = torch.full_like(parameter, float(index))
    original_modes = {name: module.training for name, module in selector.named_modules()}
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    original_gradients = [parameter.grad for parameter in parameters]
    original_gradient_values = [
        None if gradient is None else gradient.clone() for gradient in original_gradients
    ]
    original_buffers = {name: value.clone() for name, value in model.named_buffers()}
    original_weights = [parameter.detach().clone() for parameter in parameters]
    original_mask = selector.mask.clone()
    evaluation_modes = []

    def observe_evaluation(_module, _inputs):
        evaluation_modes.append([module.training for module in selector.modules()])
        assert not any(parameter.requires_grad for parameter in parameters)
        if failure == "forward":
            raise RuntimeError("model forward failed")

    def loss_fn(prediction, target):
        if failure == "loss":
            return (prediction - target).square()
        return (prediction - target).square().mean()

    inputs = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 1.0]])
    targets = torch.tensor([[1.0], [-1.0]])
    handle = model.register_forward_pre_hook(observe_evaluation)
    try:
        if failure == "forward":
            with pytest.raises(RuntimeError, match="model forward failed"):
                selector.compute_mask_derivatives([(inputs, targets)], loss_fn)
        elif failure == "loss":
            with pytest.raises(ValueError, match="loss_fn must return a scalar tensor"):
                selector.compute_mask_derivatives([(inputs, targets)], loss_fn)
        else:
            gradient, hessian = selector.compute_mask_derivatives([(inputs, targets)], loss_fn)
            assert gradient.shape == (3,)
            assert hessian.shape == (3, 3)
            assert np.isfinite(gradient).all()
            assert np.isfinite(hessian).all()
    finally:
        handle.remove()

    assert evaluation_modes == [[False] * len(original_modes)]
    assert [parameter.requires_grad for parameter in parameters] == original_requires_grad
    for parameter, original, expected, weight in zip(
        parameters, original_gradients, original_gradient_values, original_weights
    ):
        assert parameter.grad is original
        if expected is not None:
            assert torch.equal(parameter.grad, expected)
        assert torch.equal(parameter, weight)
    for name, value in model.named_buffers():
        assert torch.equal(value, original_buffers[name])
    assert torch.equal(selector.mask, original_mask)
    assert {name: module.training for name, module in selector.named_modules()} == original_modes
