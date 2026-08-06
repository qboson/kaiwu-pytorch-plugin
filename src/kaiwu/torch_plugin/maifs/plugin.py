"""Hard-mask feature selection wrapper for PyTorch models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import ceil

import numpy as np
import torch
from torch import nn

from .qubo import qubo_objective, solve_qubo

Batch = tuple[torch.Tensor, torch.Tensor]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_CARDINALITY_PENALTY = 10.0


def _tensor_to_numpy(
    tensor: torch.Tensor,
    dtype: type | np.dtype | None = None,
) -> np.ndarray:
    """不使用 PyTorch NumPy 桥接，把张量转换为 NumPy 数组。"""
    array = np.array(tensor.detach().cpu().tolist())
    if dtype is not None:
        return array.astype(dtype)
    return array


class FeatureSelectionWrapper(nn.Module):
    """把 PyTorch 基模型包装成带特征选择 mask 的模型。

    训练时用户仍然按照普通 PyTorch 模型的方式传入数据、loss 和 optimizer。
    本类会在输入进入基模型前乘上一个 0/1 mask，并按固定周期更新 mask。

    ``FeatureSelectionWrapper`` 只提供一个统一的 ``solver_kwargs`` 参数。
    不同求解器需要的参数都放在这个字典里，主类不会为 local_search、sa、
    kaiwu_cim 分别暴露一组独立参数。

    Args:
        model (nn.Module): 被包装的 PyTorch 基模型。
        feature_dim (int): 需要做特征选择的输入特征数量。
        lambda_reg (float, optional): 选择特征数量的线性惩罚系数。默认为 0.0。
        cardinality_k (int | None, optional): 期望选择的特征数量。默认为 None。
        min_selected_features (int | None, optional): 最少保留的特征数量。默认为 None。
        max_selected_features (int | None, optional): 最多保留的特征数量。默认为 None。
        solver (str, optional): 内置求解器名称。默认为 "local_search"。
        mask_update_epochs (int | None, optional): 每隔多少个 epoch 更新一次 mask。默认为 None。
        input_feature_axis (int, optional): 输入张量中特征所在维度。默认为 -1。
        solver_kwargs (dict[str, object] | None, optional): 求解器参数字典。默认为 None。

    Returns:
        FeatureSelectionWrapper: 初始化后的 PyTorch 特征选择包装器实例。

    Raises:
        TypeError: 当 model 不是 nn.Module 时抛出。
        ValueError: 当特征数量、特征数量约束或求解器名称不合法时抛出。

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> from maifs import FeatureSelectionWrapper
        >>> x = torch.randn(16, 3)
        >>> y = 2.0 * x[:, :1] - 3.0 * x[:, 1:2]
        >>> loader = DataLoader(TensorDataset(x, y), batch_size=16)
        >>> model = nn.Linear(3, 1)
        >>> selector = FeatureSelectionWrapper(
        ...     model,
        ...     feature_dim=3,
        ...     cardinality_k=2,
        ...     solver="local_search",
        ...     solver_kwargs={"max_iter": 200},
        ...     mask_update_epochs=5,
        ... )
        >>> optimizer = torch.optim.SGD(selector.model.parameters(), lr=0.1)
        >>> selector.fit_weights(loader, nn.MSELoss(), optimizer, epochs=1)
        None
    """

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        lambda_reg: float = 0.0,
        cardinality_k: int | None = None,
        min_selected_features: int | None = None,
        max_selected_features: int | None = None,
        solver: str = "local_search",
        mask_update_epochs: int | None = None,
        input_feature_axis: int = -1,
        solver_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        explicit_min = min_selected_features is not None
        feature_dim = int(feature_dim)
        max_selected_features = (
            feature_dim
            if max_selected_features is None
            else int(max_selected_features)
        )
        min_selected_features = (
            min(max(1, ceil(0.2 * feature_dim)), max_selected_features)
            if min_selected_features is None
            else int(min_selected_features)
        )

        self.model = model
        self.feature_dim = feature_dim
        self.lambda_reg = float(lambda_reg)
        self.cardinality_k = cardinality_k
        self.min_selected_features = min_selected_features
        self.max_selected_features = max_selected_features
        self.solver = str(solver)
        self.solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
        self.mask_update_epochs = (
            None if mask_update_epochs is None else int(mask_update_epochs)
        )
        self.input_feature_axis = int(input_feature_axis)
        self._min_selected_features_explicit = explicit_min
        self._trained_epochs = 0
        self.register_buffer("mask", torch.ones(self.feature_dim))

    def _mask_view(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """把一维 mask 变形成可以和输入张量广播相乘的形状。"""
        if x.ndim == 0:
            raise ValueError("x must have at least one dimension")
        axis = self.input_feature_axis % x.ndim
        if x.shape[axis] != self.feature_dim:
            raise ValueError(
                f"input feature axis has size {x.shape[axis]}, expected {self.feature_dim}"
            )
        shape = [1] * x.ndim
        shape[axis] = self.feature_dim
        return mask.reshape(tuple(shape)).to(device=x.device, dtype=x.dtype)

    def apply_mask(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """对输入张量应用当前或指定的特征 mask。"""
        resolved_mask = self.mask if mask is None else mask
        return x * self._mask_view(x, resolved_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行带特征选择 mask 的前向传播。"""
        return self.model(self.apply_mask(x))

    def fit_weights(
        self,
        data_loader: Iterable[Batch],
        loss_fn: LossFunction,
        optimizer: torch.optim.Optimizer,
        epochs: int = 1,
    ) -> float:
        """训练基模型权重，并可按固定周期更新特征 mask。"""
        epochs = int(epochs)
        self.train()
        total_loss = 0.0
        batch_count = 0

        for _ in range(epochs):
            epoch_batches = 0
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.mask.device)
                y_batch = y_batch.to(self.mask.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self(x_batch), y_batch)
                if loss.ndim != 0:
                    raise ValueError("loss_fn must return a scalar tensor")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                batch_count += 1
                epoch_batches += 1
            if epoch_batches == 0:
                raise ValueError("data_loader produced no batches")
            self._trained_epochs += 1
            if (
                self.mask_update_epochs is not None
                and self.mask_update_epochs > 0
                and self._trained_epochs % self.mask_update_epochs == 0
            ):
                self.update_mask(data_loader, loss_fn)

        if batch_count == 0:
            raise ValueError("data_loader produced no batches")
        return total_loss / batch_count

    def compute_mask_derivatives(
        self,
        data_loader: Iterable[Batch],
        loss_fn: LossFunction,
        hessian_mode: str = "full",
        max_samples: int | None = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算 loss 对连续 mask 的梯度和 Hessian。"""
        if hessian_mode not in {"full", "diagonal"}:
            raise ValueError("hessian_mode must be 'full' or 'diagonal'")

        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        count = 0
        for x_batch, y_batch in data_loader:
            xs.append(x_batch)
            ys.append(y_batch)
            count += len(x_batch)
            if max_samples is not None and count >= max_samples:
                break

        if not xs:
            raise ValueError("data_loader produced no batches")
        x_all = torch.cat(xs, dim=0)
        y_all = torch.cat(ys, dim=0)
        if max_samples is not None:
            x_all = x_all[:max_samples]
            y_all = y_all[:max_samples]
        x_all = x_all.to(self.mask.device)
        y_all = y_all.to(self.mask.device)
        was_training = self.training
        original_requires_grad = [p.requires_grad for p in self.model.parameters()]
        self.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        try:
            s = self.mask.detach().clone().to(dtype=x_all.dtype).requires_grad_(True)
            loss = loss_fn(self.model(self.apply_mask(x_all, s)), y_all)
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            gradient = torch.autograd.grad(loss, s, create_graph=True)[0]
            rows = [
                self._hessian_row(gradient, s, index, hessian_mode)
                for index in range(self.feature_dim)
            ]
            hessian = torch.stack(rows)
        finally:
            for parameter, requires_grad in zip(
                self.model.parameters(),
                original_requires_grad,
            ):
                parameter.requires_grad_(requires_grad)
            self.train(was_training)

        hessian = 0.5 * (hessian + hessian.T)
        return (
            _tensor_to_numpy(gradient, dtype=float),
            _tensor_to_numpy(hessian, dtype=float),
        )

    @staticmethod
    def _hessian_row(
        gradient: torch.Tensor,
        mask: torch.Tensor,
        index: int,
        hessian_mode: str,
    ) -> torch.Tensor:
        """计算 Hessian 的一行"""
        if gradient[index].requires_grad:
            row = torch.autograd.grad(
                gradient[index],
                mask,
                retain_graph=True,
                allow_unused=True,
            )[0]
        else:
            row = None
        if row is None:
            row = torch.zeros_like(mask)
        if hessian_mode == "diagonal":
            diagonal_row = torch.zeros_like(row)
            diagonal_row[index] = row[index]
            return diagonal_row
        return row

    def _build_qubo(
        self,
        gradient: np.ndarray,
        hessian: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """根据 mask 导数构造 QUBO 的二次项和一次项。"""
        current = _tensor_to_numpy(self.mask, dtype=float)
        q = np.asarray(hessian, dtype=float).copy()
        c = np.asarray(gradient, dtype=float) - q @ current + self.lambda_reg
        if self.cardinality_k is not None:
            q += 2.0 * _CARDINALITY_PENALTY * np.ones_like(q)
            c -= 2.0 * _CARDINALITY_PENALTY * self.cardinality_k
        return q, c

    def _project_selected_feature_count(
        self,
        candidate_mask: np.ndarray,
        q: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """把候选 mask 修正到特征数量上下界内。"""
        projected = np.asarray(candidate_mask, dtype=int).copy()
        if projected.shape != (self.feature_dim,):
            raise ValueError(f"candidate_mask must have shape ({self.feature_dim},)")
        if not np.all((projected == 0) | (projected == 1)):
            raise ValueError("candidate_mask must contain only binary 0/1 values")

        def objective_after_change(index: int, value: int) -> float:
            changed = projected.copy()
            changed[index] = value
            return qubo_objective(changed, q, c)

        selected_count = int(projected.sum())
        while selected_count > self.max_selected_features:
            selected_indices = np.flatnonzero(projected)
            drop_index = min(
                selected_indices,
                key=lambda index: (
                    objective_after_change(int(index), 0),
                    int(index),
                ),
            )
            projected[int(drop_index)] = 0
            selected_count -= 1

        enforce_minimum = self._min_selected_features_explicit or selected_count == 0
        while enforce_minimum and selected_count < self.min_selected_features:
            unselected_indices = np.flatnonzero(projected == 0)
            add_index = min(
                unselected_indices,
                key=lambda index: (
                    objective_after_change(int(index), 1),
                    int(index),
                ),
            )
            projected[int(add_index)] = 1
            selected_count += 1

        return projected

    def update_mask(
        self,
        data_loader: Iterable[Batch],
        loss_fn: LossFunction,
        hessian_mode: str = "full",
        max_samples: int | None = 1000,
    ) -> np.ndarray:
        """计算并写回一次新的特征 mask。"""
        gradient, hessian = self.compute_mask_derivatives(
            data_loader,
            loss_fn,
            hessian_mode,
            max_samples,
        )
        q, c = self._build_qubo(gradient, hessian)
        current = _tensor_to_numpy(self.mask, dtype=int)
        selected = solve_qubo(q, c, current, self.solver, **self.solver_kwargs)
        selected = self._project_selected_feature_count(selected, q, c)
        value = torch.tensor(
            np.asarray(selected).tolist(),
            device=self.mask.device,
            dtype=self.mask.dtype,
        )
        with torch.no_grad():
            self.mask.copy_(value)
        return selected.copy()

    def get_support(self) -> np.ndarray:
        """返回当前 mask 的布尔形式。"""
        return _tensor_to_numpy(self.mask, dtype=bool)

    def selected_indices(self) -> np.ndarray:
        """返回当前被选中特征的索引。 """
        return np.flatnonzero(self.get_support())

    def num_selected(self) -> int:
        """返回当前被选中的特征数量。"""
        return int(self.mask.detach().sum())
