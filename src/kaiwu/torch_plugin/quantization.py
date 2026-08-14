"""QUBO-based PyTorch post-training quantization."""

# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-positional-arguments

from __future__ import annotations

import copy
import fnmatch
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import kaiwu as kw
import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_functional


@dataclass
class _Problem:
    """Store one QUBO subproblem and its rounding-variable metadata.

    Args:
        name (str): Subproblem name used in reports and CIM task names.
        qubo_matrix (List[List[float]]): QUBO coefficient matrix.
        keys (List[Tuple[Any, ...]]): Weight indices controlled by variables.
        lower_q (List[int]): Integer value selected when a variable is 0.
        upper_q (List[int]): Integer value selected when a variable is 1.
    """

    name: str
    qubo_matrix: List[List[float]]
    keys: List[Tuple[Any, ...]]
    lower_q: List[int]
    upper_q: List[int]


class quantization(nn.Module):
    """Convert a trained PyTorch model with QUBO-based PTQ.

    The converter collects calibration activations or consumes layerwise Hessian
    matrices, builds independent QUBO subproblems for every quantizable
    ``nn.Conv2d`` and ``nn.Linear`` weight tensor, solves the binary rounding
    decisions with Kaiwu SA or CIM, and writes dequantized weights back into a
    copied model. It is a PTQ conversion tool, not a ``torch.optim.Optimizer``.

    Args:
        model (nn.Module): Trained PyTorch model to quantize.
        bits (int, optional): Signed symmetric weight bit width. ``bits=3`` maps
            weights to integer values in ``[-3, 3]`` before dequantization.
        solvers (Sequence[Mapping[str, Any]], optional): Kaiwu solver configs,
            for example ``[{"name": "cim", "params": {...}}]``.
        layers (str | Sequence[str], optional): Layer names or wildcard patterns
            to quantize. ``None`` means all Conv2d and Linear layers.
        qubo_size (int, optional): Maximum variables per QUBO chunk. Defaults to
            64.
        inplace (bool, optional): Modify ``model`` directly when True. Defaults
            to False.

    Returns:
        quantization: Converter instance whose ``activation`` and ``hessian``
        methods return ``(quantized_model, report)``.

    Raises:
        KeyError: A quantizable layer is missing calibration data or Hessian.
        Exception: Kaiwu solver exceptions are propagated.

    Examples:
        >>> converter = quantization(model, bits=3, solvers=[{"name": "sa"}])
        >>> quantized_model, report = converter.activation(calibration_data)
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 3,
        solvers: Optional[Sequence[Mapping[str, Any]]] = None,
        layers: Optional[Union[str, Sequence[str]]] = None,
        qubo_size: int = 64,
        inplace: bool = False,
    ):
        """Initialize the converter and quantization range.

        Args:
            model (nn.Module): Trained PyTorch model.
            bits (int, optional): Signed symmetric weight bit width.
            solvers (Sequence[Mapping[str, Any]], optional): Solver configs.
            layers (str | Sequence[str], optional): Layer names or patterns.
            qubo_size (int, optional): Maximum variables in one QUBO chunk.
            inplace (bool, optional): Whether to modify ``model`` directly.
        """

        super().__init__()
        self.model = model if inplace else copy.deepcopy(model)
        self.bits = bits
        self.solvers = list(solvers or [{"name": "sa", "params": {}}])
        self.layers = [layers] if isinstance(layers, str) else list(layers or [])
        self.qubo_size = qubo_size
        self.qmin = -((1 << (bits - 1)) - 1)
        self.qmax = (1 << (bits - 1)) - 1

    def activation(self, calibration_data: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize weights with calibration activations.

        Args:
            calibration_data (Any): Tensor, batch, or loader for calibration.

        Returns:
            Tuple[nn.Module, Dict[str, Any]]: Quantized model and report.
        """

        return self._convert("activation", self._capture_inputs(calibration_data))

    def hessian(self, hessians: Mapping[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize weights with layerwise Hessian matrices.

        Args:
            hessians (Mapping[str, Any]): Layer name to Hessian matrix.

        Returns:
            Tuple[nn.Module, Dict[str, Any]]: Quantized model and report.
        """

        return self._convert("hessian", hessians)

    def forward(self, data: Any, method: str = "activation") -> Tuple[nn.Module, Dict[str, Any]]:
        """Dispatch to one PTQ method.

        Args:
            data (Any): Calibration data or Hessian mapping.
            method (str, optional): Quantization method name.

        Returns:
            Tuple[nn.Module, Dict[str, Any]]: Quantized model and report.

        Raises:
            ValueError: If ``method`` is not supported.
        """

        if method == "activation":
            return self.activation(data)
        if method == "hessian":
            return self.hessian(data)
        raise ValueError('method must be "activation" or "hessian"')

    def _convert(self, method: str, data: Mapping[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Run QUBO PTQ and build a layer report.

        Args:
            method (str): Quantization method name.
            data (Mapping[str, Any]): Captured activations or Hessians.

        Returns:
            Tuple[nn.Module, Dict[str, Any]]: Quantized model and report.
        """

        layers = []
        for name, layer in self._quantizable_layers():
            problems, info = (
                self._activation_problems(name, layer, data[name])
                if method == "activation"
                else self._hessian_problems(name, layer, data[name])
            )
            integer_weight = self._nearest_weight(
                layer.weight.detach().cpu().tolist(),
                info["weight_scale"],
            )
            objectives, solver_counts, variable_count = [], {}, 0
            for problem in problems:
                binary_values, value, solver_name = self._solve(problem)
                objectives.append(value)
                solver_counts[solver_name] = solver_counts.get(solver_name, 0) + 1
                variable_count += len(binary_values)
                for key, lower_value, upper_value, bit_value in zip(
                    problem.keys,
                    problem.lower_q,
                    problem.upper_q,
                    binary_values,
                ):
                    cursor = integer_weight
                    for index in key[1:-1]:
                        cursor = cursor[int(index)]
                    cursor[int(key[-1])] = upper_value if bit_value else lower_value

            self._write_weight(layer, integer_weight, info["weight_scale"])
            layers.append(
                {
                    "name": name,
                    "method": method,
                    "bits": self.bits,
                    "weight_scale": info["weight_scale"],
                    "num_subproblems": len(objectives),
                    "num_variables": variable_count,
                    "solvers": solver_counts,
                    "objective_values": objectives,
                    "integer_weight": integer_weight,
                }
            )
        target_layers = self.layers if self.layers else "all"
        return self.model, {
            "method": method,
            "bits": self.bits,
            "target_layers": target_layers,
            "layers": layers,
        }

    def _activation_problems(
        self,
        name: str,
        layer: nn.Module,
        inputs: torch.Tensor,
    ) -> Tuple[List[_Problem], Dict[str, float]]:
        """Build activation-reconstruction QUBO chunks for one layer.

        Args:
            name (str): Layer name.
            layer (nn.Module): Quantized layer.
            inputs (torch.Tensor): Captured layer input tensor.

        Returns:
            Tuple[List[_Problem], Dict[str, float]]: QUBOs and layer metadata.
        """

        weights = layer.weight.detach().cpu()
        metric_matrix = self._activation_hessian(layer, inputs).tolist()
        weight_scale = _scale(weights.reshape(-1).tolist(), self.qmax)
        flat_weight = weights.reshape(weights.shape[0], -1).tolist()
        problems = []
        for output_index, weight_row in enumerate(flat_weight):
            keys = [
                self._weight_key(weights.shape, output_index, input_index)
                for input_index in range(len(weight_row))
            ]
            problems.extend(
                self._make_problems(
                    f"{name}.row{output_index}",
                    weight_row,
                    [weight_scale] * len(weight_row),
                    keys,
                    metric_matrix,
                )
            )
        return problems, {"weight_scale": weight_scale}

    def _hessian_problems(
        self,
        name: str,
        layer: nn.Module,
        hessian: Any,
    ) -> Tuple[List[_Problem], Dict[str, float]]:
        """Build Hessian-weighted QUBO chunks for one layer.

        Args:
            name (str): Layer name.
            layer (nn.Module): Quantized layer.
            hessian (Any): Layerwise Hessian matrix.

        Returns:
            Tuple[List[_Problem], Dict[str, float]]: QUBOs and layer metadata.
        """

        metric_matrix = (
            hessian.detach().cpu().tolist() if hasattr(hessian, "detach") else hessian
        )
        metric_matrix = (
            metric_matrix.tolist()
            if hasattr(metric_matrix, "tolist")
            else metric_matrix
        )
        weights = layer.weight.detach().cpu()
        weight_values = weights.reshape(-1).tolist()
        weight_scale = _scale(weight_values, self.qmax)
        parameter_values = [float(value) / weight_scale for value in weight_values]
        keys = [
            self._flat_weight_key(weights.shape, weight_index)
            for weight_index in range(weights.numel())
        ]
        return (
            self._make_problems(
                f"{name}.hessian",
                parameter_values,
                [1.0] * len(parameter_values),
                keys,
                metric_matrix,
            ),
            {"weight_scale": weight_scale},
        )

    def _make_problems(
        self,
        problem_prefix: str,
        parameter_values: Sequence[float],
        parameter_scales: Sequence[float],
        keys: Sequence[Tuple[Any, ...]],
        metric_matrix: Sequence[Sequence[float]],
    ) -> List[_Problem]:
        """Create chunked QUBO matrices from floor and ceil candidates.

        Args:
            problem_prefix (str): Prefix used in subproblem names.
            parameter_values (Sequence[float]): Values before rounding.
            parameter_scales (Sequence[float]): Per-value scale factors.
            keys (Sequence[Tuple[Any, ...]]): Tensor indices for variables.
            metric_matrix (Sequence[Sequence[float]]): Error metric matrix.

        Returns:
            List[_Problem]: Chunked QUBO subproblems.
        """

        problems = []
        for chunk_start in range(0, len(parameter_values), self.qubo_size):
            chunk_end = min(len(parameter_values), chunk_start + self.qubo_size)
            chunk_metric = [
                [
                    float(metric_matrix[row_index][column_index])
                    for column_index in range(chunk_start, chunk_end)
                ]
                for row_index in range(chunk_start, chunk_end)
            ]
            lower_offsets, rounding_steps, lower_q, upper_q = [], [], [], []
            for value, scale in zip(
                parameter_values[chunk_start:chunk_end],
                parameter_scales[chunk_start:chunk_end],
            ):
                lower_value = max(
                    self.qmin,
                    min(self.qmax, math.floor(float(value) / float(scale))),
                )
                upper_value = max(
                    self.qmin,
                    min(self.qmax, math.ceil(float(value) / float(scale))),
                )
                lower_q.append(int(lower_value))
                upper_q.append(int(upper_value))
                lower_offsets.append(lower_value * scale - float(value))
                rounding_steps.append((upper_value - lower_value) * scale)

            chunk_size = chunk_end - chunk_start
            qubo_matrix = [[0.0] * chunk_size for _ in range(chunk_size)]
            for row_index in range(chunk_size):
                linear_term = sum(
                    chunk_metric[row_index][column_index] * lower_offsets[column_index]
                    for column_index in range(chunk_size)
                )
                qubo_matrix[row_index][row_index] += (
                    2.0 * rounding_steps[row_index] * linear_term
                )
            for row_index in range(chunk_size):
                for column_index in range(chunk_size):
                    qubo_matrix[row_index][column_index] += (
                        rounding_steps[row_index]
                        * chunk_metric[row_index][column_index]
                        * rounding_steps[column_index]
                    )
            problems.append(
                _Problem(
                    f"{problem_prefix}.chunk{chunk_start // self.qubo_size}",
                    qubo_matrix,
                    list(keys[chunk_start:chunk_end]),
                    lower_q,
                    upper_q,
                )
            )
        return problems

    def _activation_hessian(self, layer: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Return the activation Gram matrix used by output-error PTQ.

        Args:
            layer (nn.Module): Layer receiving the captured inputs.
            inputs (torch.Tensor): Captured layer input tensor.

        Returns:
            torch.Tensor: Activation Gram matrix for QUBO construction.
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

    def _weight_key(
        self,
        shape: torch.Size,
        output_index: int,
        input_index: int,
    ) -> Tuple[Any, ...]:
        """Map one output-row offset to a tensor weight index.

        Args:
            shape (torch.Size): Weight tensor shape.
            output_index (int): Output channel or row index.
            input_index (int): Flattened input index within the output row.

        Returns:
            Tuple[Any, ...]: Nested index path into ``layer.weight``.
        """

        if len(shape) == 2:
            return ("weight", output_index, input_index)
        kernel_size = int(shape[2] * shape[3])
        input_channel, kernel_offset = input_index // kernel_size, input_index % kernel_size
        return (
            "weight",
            output_index,
            input_channel,
            kernel_offset // int(shape[3]),
            kernel_offset % int(shape[3]),
        )

    def _flat_weight_key(self, shape: torch.Size, index: int) -> Tuple[Any, ...]:
        """Map one flattened offset to a tensor weight index.

        Args:
            shape (torch.Size): Weight tensor shape.
            index (int): Flattened weight index.

        Returns:
            Tuple[Any, ...]: Nested index path into ``layer.weight``.
        """

        if len(shape) == 2:
            return ("weight", index // int(shape[1]), index % int(shape[1]))
        per_output = int(shape[1] * shape[2] * shape[3])
        return self._weight_key(shape, index // per_output, index % per_output)

    def _solve(self, problem: _Problem) -> Tuple[List[int], float, str]:
        """Solve one QUBO with configured solvers and keep the best sample.

        Args:
            problem (_Problem): QUBO subproblem to solve.

        Returns:
            Tuple[List[int], float, str]: Binary decision, objective, and solver.
        """

        best_values: List[int] = []
        best_value = math.inf
        best_solver = "none"
        for solver_spec in self.solvers:
            solver_alias = str(
                solver_spec.get("name", solver_spec.get("solver", solver_spec.get("type", "sa")))
            ).lower()
            solver_params = dict(solver_spec.get("params", solver_spec.get("parameters", {})))
            is_cim = "cim" in solver_alias.replace("_", "").replace(".", "")
            binary_values = _solve_cim(problem, solver_params) if is_cim else _solve_sa(
                problem,
                solver_params,
            )
            value = _qubo_energy(problem.qubo_matrix, binary_values)
            solver_name = (
                "kw.cim.CIMOptimizer"
                if is_cim
                else "kw.classical.SimulatedAnnealingOptimizer"
            )
            if value < best_value:
                best_values, best_value, best_solver = binary_values, value, solver_name
        return best_values, best_value, best_solver

    def _capture_inputs(self, data: Any) -> Dict[str, torch.Tensor]:
        """Collect one calibration input tensor for every quantizable layer.

        Args:
            data (Any): Tensor, batch, or loader for calibration.

        Returns:
            Dict[str, torch.Tensor]: Layer name to captured input tensor.
        """

        captured: Dict[str, torch.Tensor] = {}
        hooks = []

        def save_input(layer_name: str):
            """Create a hook that stores a layer input tensor.

            Args:
                layer_name (str): Name of the layer owning the hook.

            Returns:
                Callable: Forward hook for the selected layer.
            """

            def hook(_module, hook_inputs, _output):
                """Record the first input tensor seen by the hook.

                Args:
                    _module (nn.Module): Hooked module, unused.
                    hook_inputs (Tuple[Any, ...]): Forward inputs.
                    _output (Any): Forward output, unused.
                """

                captured.setdefault(layer_name, hook_inputs[0].detach().cpu())

            return hook

        for name, layer in self._quantizable_layers():
            hooks.append(layer.register_forward_hook(save_input(name)))

        calibration_batch = data
        if not hasattr(calibration_batch, "detach") and not (
            isinstance(calibration_batch, (list, tuple))
            and calibration_batch
            and hasattr(calibration_batch[0], "detach")
        ):
            calibration_batch = next(iter(data))
        model_input = (
            calibration_batch[0]
            if isinstance(calibration_batch, (list, tuple))
            else calibration_batch
        )
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

    def _quantizable_layers(self) -> Iterator[Tuple[str, nn.Module]]:
        """Yield selected Conv2d and Linear layers from the working model.

        Yields:
            Tuple[str, nn.Module]: Layer name and module.
        """

        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)) and self._layer_selected(name):
                yield name, layer

    def _layer_selected(self, name: str) -> bool:
        """Return whether a layer name matches the requested targets.

        Args:
            name (str): Candidate layer name.

        Returns:
            bool: True when the layer should be quantized.
        """

        return not self.layers or any(fnmatch.fnmatchcase(name, pattern) for pattern in self.layers)

    def _nearest_integer(self, value: float, scale: float) -> int:
        """Round one value into the configured integer range.

        Args:
            value (float): Floating-point weight value.
            scale (float): Quantization scale.

        Returns:
            int: Clamped signed integer value.
        """

        return max(self.qmin, min(self.qmax, round(float(value) / float(scale))))

    def _nearest_weight(self, values, scale: float):
        """Round a nested weight tensor into integer values.

        Args:
            values (Any): Scalar or nested list of weights.
            scale (float): Quantization scale.

        Returns:
            Any: Scalar or nested list of integer weights.
        """

        if isinstance(values, list):
            return [self._nearest_weight(value, scale) for value in values]
        return self._nearest_integer(float(values), scale)

    def _write_weight(self, layer: nn.Module, integer_weight, scale: float) -> None:
        """Write dequantized integer weights back to one layer.

        Args:
            layer (nn.Module): Layer receiving quantized weights.
            integer_weight (Any): Nested integer weight values.
            scale (float): Quantization scale.
        """

        weight = (
            torch.tensor(integer_weight, dtype=layer.weight.dtype, device=layer.weight.device)
            * scale
        )
        layer.weight.data.copy_(weight)


def _solve_sa(problem: _Problem, solver_params: Dict[str, Any]) -> List[int]:
    """Solve one QUBO with Kaiwu simulated annealing.

    Args:
        problem (_Problem): QUBO subproblem to solve.
        solver_params (Dict[str, Any]): Simulated annealing parameters.

    Returns:
        List[int]: Binary rounding decisions.
    """

    config = {
        "initial_temperature": 100,
        "alpha": 0.99,
        "cutoff_temperature": 0.001,
        "iterations_per_t": 100,
        "size_limit": 1,
    }
    config.update(solver_params)
    qubo_model = kw.qubo.qubo_matrix_to_qubo_model(np.array(problem.qubo_matrix, dtype=float))
    sample, _objective = kw.solver.SimpleSolver(
        kw.classical.SimulatedAnnealingOptimizer(**config)
    ).solve_qubo(qubo_model)
    binary_values = [0] * len(problem.qubo_matrix)
    for key, value in sample.items():
        match = re.search(r"\[(\d+)\]", str(key))
        if match:
            binary_values[int(match.group(1))] = int(round(float(value)))
    return binary_values


def _solve_cim(problem: _Problem, solver_params: Dict[str, Any]) -> List[int]:
    """Solve one QUBO through Kaiwu CIMOptimizer after Ising conversion.

    Args:
        problem (_Problem): QUBO subproblem to solve.
        solver_params (Dict[str, Any]): CIM optimizer parameters.

    Returns:
        List[int]: Binary rounding decisions with the best QUBO energy.
    """

    base_task_name = solver_params.pop("task_name", "qubo_ptq")
    bit_width = solver_params.pop("bit_width", 14)
    save_dir = Path(solver_params.pop("save_dir", Path.cwd() / "kaiwu_cim_results"))
    save_dir.mkdir(parents=True, exist_ok=True)
    kw.common.CheckpointManager.save_dir = str(save_dir.resolve())
    solver_params["task_name"] = _safe_name(f"{base_task_name}_{problem.name}")
    ising_matrix, _ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(
        np.array(problem.qubo_matrix, dtype=float)
    )
    raw_size = ising_matrix.shape[0]
    if bit_width is not None:
        ising_matrix = kw.ising.adjust_ising_matrix_precision(ising_matrix, bit_width=bit_width)
    samples = kw.cim.CIMOptimizer(**solver_params).solve(ising_matrix)
    sample_rows = samples.tolist() if hasattr(samples, "tolist") else list(samples)
    if sample_rows and not isinstance(sample_rows[0], list):
        sample_rows = [sample_rows]

    best_bits, best_value = None, math.inf
    for sample_row in sample_rows:
        if raw_size == len(problem.qubo_matrix) + 1:
            auxiliary_spin = int(sample_row[len(problem.qubo_matrix)])
            candidates = [
                [
                    int((1 - int(sample_row[variable_index]) * auxiliary_spin) // 2)
                    for variable_index in range(len(problem.qubo_matrix))
                ],
                [
                    int((1 + int(sample_row[variable_index]) * auxiliary_spin) // 2)
                    for variable_index in range(len(problem.qubo_matrix))
                ],
            ]
        else:
            candidates = [
                [
                    int((1 + int(sample_row[variable_index])) // 2)
                    for variable_index in range(len(problem.qubo_matrix))
                ],
                [
                    int((1 - int(sample_row[variable_index])) // 2)
                    for variable_index in range(len(problem.qubo_matrix))
                ],
            ]
        for binary_values in candidates:
            value = _qubo_energy(problem.qubo_matrix, binary_values)
            if value < best_value:
                best_bits, best_value = binary_values, value
    return best_bits or [0] * len(problem.qubo_matrix)


def _qubo_energy(qubo_matrix: Sequence[Sequence[float]], binary_values: Sequence[int]) -> float:
    """Evaluate one QUBO objective value.

    Args:
        qubo_matrix (Sequence[Sequence[float]]): QUBO coefficient matrix.
        binary_values (Sequence[int]): Binary sample to evaluate.

    Returns:
        float: QUBO energy for the sample.
    """

    return sum(
        qubo_matrix[row_index][column_index]
        for row_index, binary_row_value in enumerate(binary_values)
        if binary_row_value
        for column_index, binary_column_value in enumerate(binary_values)
        if binary_column_value
    )


def _scale(values: Sequence[float], qmax: int) -> float:
    """Compute a symmetric quantization scale.

    Args:
        values (Sequence[float]): Floating-point values to quantize.
        qmax (int): Positive maximum integer value.

    Returns:
        float: Scale mapping integers back to floating-point values.
    """

    max_abs = max([abs(float(value)) for value in values] or [0.0])
    return 1.0 if max_abs == 0.0 else max_abs / float(qmax)


def _safe_name(name: str) -> str:
    """Sanitize text for a Kaiwu task name.

    Args:
        name (str): Raw task name.

    Returns:
        str: Kaiwu-compatible task name.
    """

    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("._-")[:80] or "qubo_ptq"
