"""Hard-mask feature selection wrapper for PyTorch models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import ceil

import numpy as np
import torch
from torch import nn

from .qubo import solve_qubo

Batch = tuple[torch.Tensor, torch.Tensor]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_CARDINALITY_PENALTY = 10.0


def _tensor_to_numpy(
    tensor: torch.Tensor,
    dtype: type | np.dtype | None = None,
) -> np.ndarray:
    """Convert a tensor to a NumPy array without using the PyTorch NumPy bridge.

    Args:
        tensor: Tensor to convert.
        dtype: Optional NumPy-compatible dtype for the returned array.

    Returns:
        The tensor values as a NumPy array.
    """
    array = np.array(tensor.detach().cpu().tolist())
    if dtype is not None:
        return array.astype(dtype)
    return array


class FeatureSelectionWrapper(nn.Module):
    """Wrap a PyTorch model with a hard 0/1 feature-selection mask.

    Args:
        model: PyTorch model that receives masked inputs.
        feature_dim: Number of input features controlled by the mask.
        lambda_reg: Linear penalty applied to selected features.
        cardinality_k: Optional target number of selected features.
        min_selected_features: Optional lower bound for selected features.
        max_selected_features: Optional upper bound for selected features.
        solver: Built-in solver name, such as ``"local_search","sa","kaiwu_cim"``.
        mask_update_epochs: Optional number of epochs between mask updates.
        input_feature_axis: Axis that contains the selectable features.
        solver_kwargs: Optional keyword arguments passed to the solver.

    Returns:
        FeatureSelectionWrapper: Initialized PyTorch feature-selection wrapper.

    Raises:
        TypeError: If ``model`` is not an ``nn.Module``.
        ValueError: If feature counts, feature-count bounds, or solver names are
            invalid.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> from maifs import FeatureSelectionWrapper
        >>> input_feature = torch.randn(16, 3)
        >>> target = 2.0 * input_feature[:, :1] - 3.0 * input_feature[:, 1:2]
        >>> loader = DataLoader(TensorDataset(input_feature, target), batch_size=16)
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
        >>> selector.fit_weights(loader, nn.MSELoss(), optimizer, train_epochs=1)
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

    def _mask_view(
        self,
        input_feature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reshape a one-dimensional mask for broadcasting over an input tensor.

        Args:
            input_feature: Input tensor that will be multiplied by the mask.
            mask: One-dimensional feature mask.

        Returns:
            The mask reshaped and typed for broadcasting with ``input_feature``.

        Raises:
            ValueError: If ``input_feature`` has no dimensions or the configured
                feature axis does not match ``feature_dim``.
        """
        if input_feature.ndim == 0:
            raise ValueError("input_feature must have at least one dimension")
        axis = self.input_feature_axis % input_feature.ndim
        axis_size = input_feature.shape[axis]
        if axis_size != self.feature_dim:
            raise ValueError(
                f"input feature axis has size {axis_size}, "
                f"expected {self.feature_dim}"
            )
        shape = [1] * input_feature.ndim
        shape[axis] = self.feature_dim
        return mask.reshape(tuple(shape)).to(
            device=input_feature.device,
            dtype=input_feature.dtype,
        )

    def apply_mask(
        self,
        input_feature: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the current or provided feature mask to an input tensor.

        Args:
            input_feature: Input tensor to mask.
            mask: Optional mask to use instead of ``self.mask``.

        Returns:
            The masked input tensor.
        """
        resolved_mask = self.mask if mask is None else mask
        return input_feature * self._mask_view(input_feature, resolved_mask)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with the feature-selection mask applied.

        Args:
            input_feature: Input tensor for the wrapped model.

        Returns:
            The wrapped model output for the masked input.
        """
        return self.model(self.apply_mask(input_feature))

    def fit_weights(
        self,
        data_loader: Iterable[Batch],
        loss_fn: LossFunction,
        optimizer: torch.optim.Optimizer,
        train_epochs: int = 1,
    ) -> float:
        """Train wrapped-model weights and optionally refresh the feature mask.

        Args:
            data_loader: Iterable that yields ``(input_batch, target_batch)``.
            loss_fn: Loss function returning a scalar tensor.
            optimizer: Optimizer for the wrapped model parameters.
            train_epochs: Number of training epochs to run.

        Returns:
            Mean loss over all processed batches.

        Raises:
            ValueError: If ``data_loader`` yields no batches or ``loss_fn`` does
                not return a scalar tensor.
        """
        train_epochs = int(train_epochs)
        self.train()
        total_loss = 0.0
        batch_count = 0

        for _ in range(train_epochs):
            epoch_batches = 0
            for input_batch, target_batch in data_loader:
                input_batch = input_batch.to(self.mask.device)
                target_batch = target_batch.to(self.mask.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self(input_batch), target_batch)
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
        """Compute the loss gradient and Hessian with respect to a soft mask.

        Args:
            data_loader: Iterable that yields ``(input_batch, target_batch)``.
            loss_fn: Loss function returning a scalar tensor.
            hessian_mode: ``"full"`` for the full Hessian or ``"diagonal"`` for
                diagonal-only Hessian rows.
            max_samples: Optional maximum number of samples used for derivatives.

        Returns:
            A tuple containing the mask gradient and Hessian as NumPy arrays.

        Raises:
            ValueError: If ``hessian_mode`` is unsupported, ``data_loader`` yields
                no batches, or ``loss_fn`` does not return a scalar tensor.
        """
        if hessian_mode not in {"full", "diagonal"}:
            raise ValueError("hessian_mode must be 'full' or 'diagonal'")

        input_batches: list[torch.Tensor] = []
        target_batches: list[torch.Tensor] = []
        count = 0
        for input_batch, target_batch in data_loader:
            input_batches.append(input_batch)
            target_batches.append(target_batch)
            count += len(input_batch)
            if max_samples is not None and count >= max_samples:
                break

        if not input_batches:
            raise ValueError("data_loader produced no batches")
        input_all = torch.cat(input_batches, dim=0)
        target_all = torch.cat(target_batches, dim=0)
        if max_samples is not None:
            input_all = input_all[:max_samples]
            target_all = target_all[:max_samples]
        input_all = input_all.to(self.mask.device)
        target_all = target_all.to(self.mask.device)
        was_training = self.training
        original_requires_grad = [p.requires_grad for p in self.model.parameters()]
        self.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        try:
            continuous_mask = (
                self.mask.detach().clone().to(dtype=input_all.dtype).requires_grad_(True)
            )
            loss = loss_fn(
                self.model(self.apply_mask(input_all, continuous_mask)),
                target_all,
            )
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            gradient = torch.autograd.grad(
                loss,
                continuous_mask,
                create_graph=True,
            )[0]
            rows = [
                self._hessian_row(gradient, continuous_mask, index, hessian_mode)
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
        continuous_mask: torch.Tensor,
        index: int,
        hessian_mode: str,
    ) -> torch.Tensor:
        """Compute one row of the Hessian matrix.

        Args:
            gradient: Gradient tensor whose entries are differentiated.
            continuous_mask: Differentiable mask tensor.
            index: Hessian row index to compute.
            hessian_mode: ``"full"`` or ``"diagonal"``.

        Returns:
            The requested Hessian row.
        """
        if gradient[index].requires_grad:
            row = torch.autograd.grad(
                gradient[index],
                continuous_mask,
                retain_graph=True,
                allow_unused=True,
            )[0]
        else:
            row = None
        if row is None:
            row = torch.zeros_like(continuous_mask)
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
        """Build QUBO quadratic and linear terms from mask derivatives.

        Args:
            gradient: Loss gradient with respect to the mask.
            hessian: Loss Hessian with respect to the mask.

        Returns:
            A tuple containing the quadratic matrix and linear vector.
        """
        current_mask = _tensor_to_numpy(self.mask, dtype=float)
        quadratic_matrix = np.asarray(hessian, dtype=float).copy()
        linear_vector = (
            np.asarray(gradient, dtype=float)
            - quadratic_matrix @ current_mask
            + self.lambda_reg
        )
        if self.cardinality_k is not None:
            quadratic_matrix += (
                2.0 * _CARDINALITY_PENALTY * np.ones_like(quadratic_matrix)
            )
            linear_vector -= 2.0 * _CARDINALITY_PENALTY * self.cardinality_k
        return quadratic_matrix, linear_vector

    def _project_selected_feature_count(
        self,
        candidate_mask: np.ndarray,
        quadratic_matrix: np.ndarray,
        linear_vector: np.ndarray,
    ) -> np.ndarray:
        """Project a candidate mask into the configured feature-count bounds.

        Args:
            candidate_mask: Binary mask proposed by the QUBO solver.
            quadratic_matrix: QUBO quadratic term.
            linear_vector: QUBO linear term.

        Returns:
            A binary mask whose selected-feature count satisfies the bounds.

        Raises:
            ValueError: If ``candidate_mask`` has the wrong shape or contains
                non-binary values.
        """
        projected = np.asarray(candidate_mask, dtype=int).copy()
        if projected.shape != (self.feature_dim,):
            raise ValueError(f"candidate_mask must have shape ({self.feature_dim},)")
        if not np.all((projected == 0) | (projected == 1)):
            raise ValueError("candidate_mask must contain only binary 0/1 values")

        def objective_after_change(index: int, value: int) -> float:
            """Evaluate the QUBO objective after flipping one projected bit.

            Args:
                index: Candidate mask index to change.
                value: Binary value to assign at ``index``.

            Returns:
                Objective value after the candidate change.
            """
            changed = projected.copy()
            changed[index] = value
            return float(
                0.5 * changed @ quadratic_matrix @ changed
                + linear_vector @ changed
            )

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
        """Compute and store one updated feature mask.

        Args:
            data_loader: Iterable that yields ``(input_batch, target_batch)``.
            loss_fn: Loss function returning a scalar tensor.
            hessian_mode: ``"full"`` for the full Hessian or ``"diagonal"`` for
                diagonal-only Hessian rows.
            max_samples: Optional maximum number of samples used for derivatives.

        Returns:
            The newly selected binary mask as a NumPy array.

        Raises:
            ValueError: If derivative computation or projection validation fails.
            RuntimeError: If the configured QUBO solver fails.
        """
        gradient, hessian = self.compute_mask_derivatives(
            data_loader,
            loss_fn,
            hessian_mode,
            max_samples,
        )
        quadratic_matrix, linear_vector = self._build_qubo(gradient, hessian)
        current_mask = _tensor_to_numpy(self.mask, dtype=int)
        selected = solve_qubo(
            quadratic_matrix,
            linear_vector,
            current_mask,
            self.solver,
            **self.solver_kwargs,
        )
        selected = self._project_selected_feature_count(
            selected,
            quadratic_matrix,
            linear_vector,
        )
        value = torch.tensor(
            np.asarray(selected).tolist(),
            device=self.mask.device,
            dtype=self.mask.dtype,
        )
        with torch.no_grad():
            self.mask.copy_(value)
        return selected.copy()

    def get_support(self) -> np.ndarray:
        """Return the current feature mask as booleans.

        Returns:
            Boolean array where selected features are ``True``.
        """
        return _tensor_to_numpy(self.mask, dtype=bool)

    def selected_indices(self) -> np.ndarray:
        """Return the indices of currently selected features.

        Returns:
            One-dimensional array of selected feature indices.
        """
        return np.flatnonzero(self.get_support())

    def num_selected(self) -> int:
        """Return the number of currently selected features.

        Returns:
            Number of selected features in the current mask.
        """
        return int(self.mask.detach().sum())
