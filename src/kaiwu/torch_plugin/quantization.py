"""QUBO-based PyTorch post-training quantization."""

# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-positional-arguments

from __future__ import annotations

import copy
import fnmatch
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_functional

import kaiwu as kw


@dataclass
class _Problem:
    """Represent one chunked QUBO rounding problem.

    Args:
        name (str): Name used in reports and CIM tasks.
        qubo_matrix (list[list[float]]): QUBO coefficient matrix.
        indices (list[int]): Flat weight indices controlled by variables.
        lower_values (list[int]): Integer values selected by binary 0.
        upper_values (list[int]): Integer values selected by binary 1.
    """

    name: str
    qubo_matrix: list[list[float]]
    indices: list[int]
    lower_values: list[int]
    upper_values: list[int]


class quantization(nn.Module):
    """Convert a trained PyTorch model with QUBO-based PTQ.

    The converter supports two methods: ``activation`` builds QUBOs from
    calibration activations, while ``hessian`` builds QUBOs from layerwise
    Hessian matrices. Both methods quantize selected ``nn.Conv2d`` and
    ``nn.Linear`` weights, then write dequantized weights into a copied model.

    Args:
        model (nn.Module): Trained PyTorch model.
        bits (int, optional): Signed symmetric weight bit width.
        solvers (Sequence[Mapping[str, Any]], optional): Solver configs in
            ``[{"name": "sa"|"cim", "params": {...}}]`` format.
        layers (str | Sequence[str] | None, optional): Layer names or wildcard
            patterns to quantize. ``None`` means all Conv2d and Linear layers.
        qubo_size (int, optional): Maximum variables in one QUBO chunk.
        inplace (bool, optional): Whether to modify ``model`` directly.

    Returns:
        quantization: Converter whose methods return ``(model, report)``.

    Examples:
        >>> converter = quantization(model, bits=4, solvers=[{"name": "sa"}])
        >>> quantized_model, report = converter.activation(calibration_images)
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 3,
        solvers: Sequence[Mapping[str, Any]] | None = None,
        layers: str | Sequence[str] | None = None,
        qubo_size: int = 64,
        inplace: bool = False,
    ):
        """Initialize model copy, quantization range, and solver configs.

        Args:
            model (nn.Module): Trained PyTorch model.
            bits (int, optional): Signed symmetric weight bit width.
            solvers (Sequence[Mapping[str, Any]] | None, optional): Solvers.
            layers (str | Sequence[str] | None, optional): Target layers.
            qubo_size (int, optional): Maximum QUBO variables per chunk.
            inplace (bool, optional): Whether to quantize in place.
        """

        super().__init__()
        self.model = model if inplace else copy.deepcopy(model)
        self.bits = bits
        self.solvers = list(solvers or [{"name": "sa", "params": {}}])
        self.layers = [layers] if isinstance(layers, str) else list(layers or [])
        self.qubo_size = qubo_size
        self.qmin = -((1 << (bits - 1)) - 1)
        self.qmax = (1 << (bits - 1)) - 1

    def activation(self, calibration_data: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Quantize weights with calibration activations.

        Args:
            calibration_data (Any): Calibration tensor or ``(tensor, label)`` batch.

        Returns:
            tuple[nn.Module, dict[str, Any]]: Quantized model and report.
        """

        return self._convert("activation", self._capture_inputs(calibration_data))

    def hessian(self, hessians: Mapping[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
        """Quantize weights with layerwise Hessian matrices.

        Args:
            hessians (Mapping[str, Any]): Layer name to Hessian matrix.

        Returns:
            tuple[nn.Module, dict[str, Any]]: Quantized model and report.
        """

        return self._convert("hessian", hessians)

    def forward(self, data: Any, method: str = "activation") -> tuple[nn.Module, dict[str, Any]]:
        """Dispatch to one PTQ method.

        Args:
            data (Any): Calibration data or Hessian mapping.
            method (str, optional): ``"activation"`` or ``"hessian"``.

        Returns:
            tuple[nn.Module, dict[str, Any]]: Quantized model and report.

        Raises:
            ValueError: If ``method`` is unsupported.
        """

        if method == "activation":
            return self.activation(data)
        if method == "hessian":
            return self.hessian(data)
        raise ValueError('method must be "activation" or "hessian"')

    def _convert(self, method: str, data: Mapping[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
        """Quantize selected layers and collect the report.

        Args:
            method (str): Quantization method name.
            data (Mapping[str, Any]): Captured activations or Hessians.

        Returns:
            tuple[nn.Module, dict[str, Any]]: Quantized model and report.
        """

        report_layers = []
        for layer_name, layer in self._quantizable_layers():
            problems, weight_scale = (
                self._activation_problems(layer_name, layer, data[layer_name])
                if method == "activation"
                else self._hessian_problems(layer_name, layer, data[layer_name])
            )
            integer_weight = torch.round(layer.weight.detach().cpu() / weight_scale)
            integer_weight = integer_weight.clamp(self.qmin, self.qmax).to(torch.int64)
            flat_integer_weight = integer_weight.reshape(-1)
            objectives, solver_counts, variable_count = [], {}, 0

            for problem in problems:
                binary_values, objective, solver_name = self._solve(problem)
                objectives.append(objective)
                solver_counts[solver_name] = solver_counts.get(solver_name, 0) + 1
                variable_count += len(binary_values)
                for flat_index, lower_value, upper_value, binary_value in zip(
                    problem.indices,
                    problem.lower_values,
                    problem.upper_values,
                    binary_values,
                ):
                    flat_integer_weight[flat_index] = (
                        upper_value if binary_value else lower_value
                    )

            with torch.no_grad():
                dequantized_weight = integer_weight.to(layer.weight.dtype) * weight_scale
                layer.weight.copy_(dequantized_weight.to(layer.weight.device))

            report_layers.append(
                {
                    "name": layer_name,
                    "method": method,
                    "bits": self.bits,
                    "weight_scale": weight_scale,
                    "num_subproblems": len(objectives),
                    "num_variables": variable_count,
                    "solvers": solver_counts,
                    "objective_values": objectives,
                    "integer_weight": integer_weight.tolist(),
                }
            )

        return self.model, {
            "method": method,
            "bits": self.bits,
            "target_layers": self.layers or "all",
            "layers": report_layers,
        }

    def _activation_problems(
        self,
        layer_name: str,
        layer: nn.Module,
        inputs: torch.Tensor,
    ) -> tuple[list[_Problem], float]:
        """Build activation-reconstruction QUBO chunks for one layer.

        Args:
            layer_name (str): Layer name.
            layer (nn.Module): Quantized Conv2d or Linear layer.
            inputs (torch.Tensor): Captured layer input tensor.

        Returns:
            tuple[list[_Problem], float]: QUBO chunks and weight scale.
        """

        weights = layer.weight.detach().cpu()
        metric_matrix = self._activation_metric(layer, inputs).tolist()
        weight_scale = _scale(weights.reshape(-1).tolist(), self.qmax)
        flat_rows = weights.reshape(weights.shape[0], -1)
        row_width = flat_rows.shape[1]
        problems = []

        for output_index, weight_row in enumerate(flat_rows):
            problems.extend(
                self._make_problems(
                    f"{layer_name}.row{output_index}",
                    weight_row.tolist(),
                    [weight_scale] * row_width,
                    int(output_index * row_width),
                    metric_matrix,
                )
            )
        return problems, weight_scale

    def _hessian_problems(
        self,
        layer_name: str,
        layer: nn.Module,
        hessian: Any,
    ) -> tuple[list[_Problem], float]:
        """Build Hessian-weighted QUBO chunks for one layer.

        Args:
            layer_name (str): Layer name.
            layer (nn.Module): Quantized Conv2d or Linear layer.
            hessian (Any): Layerwise Hessian matrix.

        Returns:
            tuple[list[_Problem], float]: QUBO chunks and weight scale.
        """

        weights = layer.weight.detach().cpu()
        weight_values = weights.reshape(-1).tolist()
        weight_scale = _scale(weight_values, self.qmax)
        normalized_values = [float(value) / weight_scale for value in weight_values]
        metric_matrix = torch.as_tensor(hessian, dtype=torch.float64).tolist()
        return (
            self._make_problems(
                f"{layer_name}.hessian",
                normalized_values,
                [1.0] * len(normalized_values),
                0,
                metric_matrix,
            ),
            weight_scale,
        )

    def _make_problems(
        self,
        problem_prefix: str,
        parameter_values: Sequence[float],
        parameter_scales: Sequence[float],
        flat_offset: int,
        metric_matrix: Sequence[Sequence[float]],
    ) -> list[_Problem]:
        """Create chunked QUBO matrices from floor and ceil candidates.

        Args:
            problem_prefix (str): Prefix for chunk names.
            parameter_values (Sequence[float]): Values before rounding.
            parameter_scales (Sequence[float]): Per-value scale factors.
            flat_offset (int): Offset into flattened layer weights.
            metric_matrix (Sequence[Sequence[float]]): Error metric matrix.

        Returns:
            list[_Problem]: Chunked QUBO subproblems.
        """

        problems = []
        for chunk_start in range(0, len(parameter_values), self.qubo_size):
            chunk_end = min(len(parameter_values), chunk_start + self.qubo_size)
            chunk_range = range(chunk_start, chunk_end)
            chunk_metric = [
                [float(metric_matrix[row_index][col_index]) for col_index in chunk_range]
                for row_index in chunk_range
            ]
            lower_offsets, rounding_steps, lower_values, upper_values = [], [], [], []

            for value, scale in zip(
                parameter_values[chunk_start:chunk_end],
                parameter_scales[chunk_start:chunk_end],
            ):
                lower_value = math.floor(float(value) / float(scale))
                upper_value = math.ceil(float(value) / float(scale))
                lower_value = max(self.qmin, min(self.qmax, lower_value))
                upper_value = max(self.qmin, min(self.qmax, upper_value))
                lower_values.append(int(lower_value))
                upper_values.append(int(upper_value))
                lower_offsets.append(lower_value * scale - float(value))
                rounding_steps.append((upper_value - lower_value) * scale)

            chunk_size = chunk_end - chunk_start
            qubo_matrix = [[0.0] * chunk_size for _ in range(chunk_size)]
            for row_index in range(chunk_size):
                linear_term = sum(
                    chunk_metric[row_index][col_index] * lower_offsets[col_index]
                    for col_index in range(chunk_size)
                )
                qubo_matrix[row_index][row_index] += (
                    2.0 * rounding_steps[row_index] * linear_term
                )
                for col_index in range(chunk_size):
                    qubo_matrix[row_index][col_index] += (
                        rounding_steps[row_index]
                        * chunk_metric[row_index][col_index]
                        * rounding_steps[col_index]
                    )

            problems.append(
                _Problem(
                    name=f"{problem_prefix}.chunk{chunk_start // self.qubo_size}",
                    qubo_matrix=qubo_matrix,
                    indices=list(range(flat_offset + chunk_start, flat_offset + chunk_end)),
                    lower_values=lower_values,
                    upper_values=upper_values,
                )
            )
        return problems

    def _activation_metric(self, layer: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Calculate the activation Gram matrix for output-error PTQ.

        Args:
            layer (nn.Module): Layer receiving the captured inputs.
            inputs (torch.Tensor): Captured layer input tensor.

        Returns:
            torch.Tensor: Activation Gram matrix.
        """

        input_tensor = inputs.detach().cpu()
        if isinstance(layer, nn.Conv2d):
            patches = torch_functional.unfold(
                input_tensor,
                layer.kernel_size,
                dilation=layer.dilation,
                padding=layer.padding,
                stride=layer.stride,
            )
            mean_patches = patches.mean(dim=0).transpose(0, 1)
            return mean_patches.transpose(0, 1).matmul(mean_patches)
        mean_input = input_tensor.reshape(-1, layer.weight.shape[1]).mean(dim=0)
        return torch.outer(mean_input, mean_input)

    def _solve(self, problem: _Problem) -> tuple[list[int], float, str]:
        """Solve one QUBO with configured solvers.

        Args:
            problem (_Problem): QUBO subproblem.

        Returns:
            tuple[list[int], float, str]: Binary values, objective, and solver.
        """

        best_values, best_objective, best_solver = [], math.inf, "none"
        for solver_spec in self.solvers:
            solver_name = str(solver_spec.get("name", "sa")).lower()
            solver_params = dict(solver_spec.get("params", {}))
            is_cim = "cim" in solver_name
            binary_values = (
                _solve_cim(problem, solver_params)
                if is_cim
                else _solve_sa(problem, solver_params)
            )
            objective = _qubo_energy(problem.qubo_matrix, binary_values)
            solver_label = (
                "kw.cim.CIMOptimizer"
                if is_cim
                else "kw.classical.SimulatedAnnealingOptimizer"
            )
            if objective < best_objective:
                best_values, best_objective, best_solver = (
                    binary_values,
                    objective,
                    solver_label,
                )
        return best_values, best_objective, best_solver

    def _capture_inputs(self, data: Any) -> dict[str, torch.Tensor]:
        """Collect layer inputs with forward hooks.

        Args:
            data (Any): Calibration tensor or ``(tensor, label)`` batch.

        Returns:
            dict[str, torch.Tensor]: Layer name to captured input tensor.
        """

        captured: dict[str, torch.Tensor] = {}
        hooks = []

        def save_input(layer_name: str):
            """Create a hook that stores the first input tensor.

            Args:
                layer_name (str): Hooked layer name.

            Returns:
                Callable: Forward hook.
            """

            def hook(_module, hook_inputs, _output):
                """Record the layer input.

                Args:
                    _module (nn.Module): Hooked module, unused.
                    hook_inputs (tuple[Any, ...]): Forward inputs.
                    _output (Any): Forward output, unused.
                """

                captured.setdefault(layer_name, hook_inputs[0].detach().cpu())

            return hook

        for layer_name, layer in self._quantizable_layers():
            hooks.append(layer.register_forward_hook(save_input(layer_name)))

        model_input = data[0] if isinstance(data, (list, tuple)) else data
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.model(model_input.to(device))
        if was_training:
            self.model.train()
        for hook_handle in hooks:
            hook_handle.remove()
        return captured

    def _quantizable_layers(self) -> Iterator[tuple[str, nn.Module]]:
        """Yield selected Conv2d and Linear layers.

        Yields:
            tuple[str, nn.Module]: Layer name and layer object.
        """

        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)) and self._layer_selected(layer_name):
                yield layer_name, layer

    def _layer_selected(self, layer_name: str) -> bool:
        """Check whether a layer should be quantized.

        Args:
            layer_name (str): Candidate layer name.

        Returns:
            bool: True when the layer matches ``layers``.
        """

        return not self.layers or any(
            fnmatch.fnmatchcase(layer_name, pattern) for pattern in self.layers
        )


def _solve_sa(problem: _Problem, solver_params: dict[str, Any]) -> list[int]:
    """Solve one QUBO with Kaiwu simulated annealing.

    Args:
        problem (_Problem): QUBO subproblem.
        solver_params (dict[str, Any]): SA optimizer parameters.

    Returns:
        list[int]: Binary rounding decisions.
    """

    config = {
        "initial_temperature": 100,
        "alpha": 0.99,
        "cutoff_temperature": 0.001,
        "iterations_per_t": 100,
        "size_limit": 1,
    }
    config.update(solver_params)
    qubo_model = kw.qubo.qubo_matrix_to_qubo_model(np.array(problem.qubo_matrix))
    sample, _objective = kw.solver.SimpleSolver(
        kw.classical.SimulatedAnnealingOptimizer(**config)
    ).solve_qubo(qubo_model)
    binary_values = [0] * len(problem.qubo_matrix)
    for key, value in sample.items():
        match = re.search(r"\[(\d+)\]", str(key))
        if match:
            binary_values[int(match.group(1))] = int(round(float(value)))
    return binary_values


def _solve_cim(problem: _Problem, solver_params: dict[str, Any]) -> list[int]:
    """Solve one QUBO with Kaiwu CIM after QUBO-to-Ising conversion.

    Args:
        problem (_Problem): QUBO subproblem.
        solver_params (dict[str, Any]): CIM optimizer parameters.

    Returns:
        list[int]: Binary rounding decisions.
    """

    base_task_name = solver_params.pop("task_name", "qubo_ptq")
    bit_width = solver_params.pop("bit_width", 14)
    save_dir = Path(solver_params.pop("save_dir", Path.cwd() / "kaiwu_cim_results"))
    save_dir.mkdir(parents=True, exist_ok=True)
    kw.common.CheckpointManager.save_dir = str(save_dir.resolve())

    task_name = f"{base_task_name}_{problem.name}"
    solver_params["task_name"] = (
        re.sub(r"[^0-9A-Za-z_.-]+", "_", task_name).strip("._-")[:80] or "qubo_ptq"
    )
    ising_matrix, _ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(
        np.array(problem.qubo_matrix)
    )
    raw_size = ising_matrix.shape[0]
    if bit_width is not None:
        ising_matrix = kw.ising.adjust_ising_matrix_precision(ising_matrix, bit_width=bit_width)

    sample_array = np.asarray(kw.cim.CIMOptimizer(**solver_params).solve(ising_matrix))
    if sample_array.ndim == 1:
        sample_array = sample_array.reshape(1, -1)

    variable_count = len(problem.qubo_matrix)
    best_values, best_objective = [0] * variable_count, math.inf
    for sample_row in sample_array.tolist():
        if raw_size == variable_count + 1:
            auxiliary_spin = int(sample_row[variable_count])
            candidates = [
                [
                    int((1 - int(sample_row[var_index]) * auxiliary_spin) // 2)
                    for var_index in range(variable_count)
                ],
                [
                    int((1 + int(sample_row[var_index]) * auxiliary_spin) // 2)
                    for var_index in range(variable_count)
                ],
            ]
        else:
            candidates = [
                [int((1 + int(sample_row[var_index])) // 2) for var_index in range(variable_count)],
                [int((1 - int(sample_row[var_index])) // 2) for var_index in range(variable_count)],
            ]
        for binary_values in candidates:
            objective = _qubo_energy(problem.qubo_matrix, binary_values)
            if objective < best_objective:
                best_values, best_objective = binary_values, objective
    return best_values


def _qubo_energy(qubo_matrix: Sequence[Sequence[float]], binary_values: Sequence[int]) -> float:
    """Evaluate one QUBO objective value.

    Args:
        qubo_matrix (Sequence[Sequence[float]]): QUBO coefficient matrix.
        binary_values (Sequence[int]): Binary sample.

    Returns:
        float: QUBO energy.
    """

    return sum(
        qubo_matrix[row_index][col_index]
        for row_index, row_value in enumerate(binary_values)
        if row_value
        for col_index, col_value in enumerate(binary_values)
        if col_value
    )


def _scale(values: Sequence[float], qmax: int) -> float:
    """Compute the symmetric quantization scale.

    Args:
        values (Sequence[float]): Floating-point weights.
        qmax (int): Positive maximum integer value.

    Returns:
        float: Weight scale.
    """

    max_abs = max([abs(float(value)) for value in values] or [0.0])
    return 1.0 if max_abs == 0.0 else max_abs / float(qmax)
