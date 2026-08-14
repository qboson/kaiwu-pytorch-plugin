"""QUBO-based PyTorch post-training quantization."""

from __future__ import annotations

import copy
import fnmatch
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import kaiwu as kw
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class _Problem:
    """Store one QUBO subproblem and its rounding-variable metadata.

    Args:
        name (str): Subproblem name used in reports and CIM task names.
        q (List[List[float]]): QUBO coefficient matrix.
        keys (List[Tuple[Any, ...]]): Weight indices controlled by variables.
        lower_q (List[int]): Integer value selected when a variable is 0.
        upper_q (List[int]): Integer value selected when a variable is 1.
    """

    name: str
    q: List[List[float]]
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
        """Initialize the converter and quantization range."""

        super().__init__()
        self.model = model if inplace else copy.deepcopy(model)
        self.bits = bits
        self.solvers = list(solvers or [{"name": "sa", "params": {}}])
        self.layers = [layers] if isinstance(layers, str) else list(layers or [])
        self.qubo_size = qubo_size
        self.qmin = -((1 << (bits - 1)) - 1)
        self.qmax = (1 << (bits - 1)) - 1

    def activation(self, calibration_data: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize Conv2d and Linear weights with calibration activations."""

        return self._convert("activation", self._capture_inputs(calibration_data))

    def hessian(self, hessians: Mapping[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize Conv2d and Linear weights with layerwise Hessians."""

        return self._convert("hessian", hessians)

    def forward(self, data: Any, method: str = "activation") -> Tuple[nn.Module, Dict[str, Any]]:
        """Dispatch to one PTQ method and return its quantized model."""

        if method == "activation":
            return self.activation(data)
        if method == "hessian":
            return self.hessian(data)
        raise ValueError('method must be "activation" or "hessian"')

    def _convert(self, method: str, data: Mapping[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Run QUBO PTQ on all quantizable layers and build a report."""

        layers = []
        for name, layer in self._quantizable_layers():
            problems, info = (
                self._activation_problems(name, layer, data[name])
                if method == "activation"
                else self._hessian_problems(name, layer, data[name])
            )
            q_weight = self._nearest_weight(layer.weight.detach().cpu().tolist(), info["weight_scale"])
            objectives, solver_counts, variable_count = [], {}, 0
            for problem in problems:
                bits, value, solver_name = self._solve(problem)
                objectives.append(value)
                solver_counts[solver_name] = solver_counts.get(solver_name, 0) + 1
                variable_count += len(bits)
                for key, lo, hi, bit in zip(problem.keys, problem.lower_q, problem.upper_q, bits):
                    cursor = q_weight
                    for index in key[1:-1]:
                        cursor = cursor[int(index)]
                    cursor[int(key[-1])] = hi if bit else lo

            self._write_weight(layer, q_weight, info["weight_scale"])
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
                    "integer_weight": q_weight,
                }
            )
        target_layers = self.layers if self.layers else "all"
        return self.model, {"method": method, "bits": self.bits, "target_layers": target_layers, "layers": layers}

    def _activation_problems(self, name: str, layer: nn.Module, inputs: torch.Tensor) -> Tuple[List[_Problem], Dict[str, float]]:
        """Build activation-reconstruction QUBO chunks for one layer."""

        weights = layer.weight.detach().cpu()
        h = self._activation_hessian(layer, inputs).tolist()
        weight_scale = _scale(weights.reshape(-1).tolist(), self.qmax)
        flat_weight = weights.reshape(weights.shape[0], -1).tolist()
        problems = []
        for out_i, row in enumerate(flat_weight):
            keys = [self._weight_key(weights.shape, out_i, in_i) for in_i in range(len(row))]
            problems.extend(self._make_problems(f"{name}.row{out_i}", row, [weight_scale] * len(row), keys, h))
        return problems, {"weight_scale": weight_scale}

    def _hessian_problems(self, name: str, layer: nn.Module, hessian: Any) -> Tuple[List[_Problem], Dict[str, float]]:
        """Build Hessian-weighted QUBO chunks for one layer."""

        h = hessian.detach().cpu().tolist() if hasattr(hessian, "detach") else hessian
        h = h.tolist() if hasattr(h, "tolist") else h
        weights = layer.weight.detach().cpu()
        weight_values = weights.reshape(-1).tolist()
        weight_scale = _scale(weight_values, self.qmax)
        params = [float(value) / weight_scale for value in weight_values]
        keys = [self._flat_weight_key(weights.shape, i) for i in range(weights.numel())]
        return self._make_problems(f"{name}.hessian", params, [1.0] * len(params), keys, h), {"weight_scale": weight_scale}

    def _make_problems(self, prefix: str, params: Sequence[float], scales: Sequence[float], keys, h) -> List[_Problem]:
        """Create chunked QUBO matrices from floor/ceil candidates."""

        problems = []
        for start in range(0, len(params), self.qubo_size):
            end = min(len(params), start + self.qubo_size)
            sub_h = [[float(h[i][j]) for j in range(start, end)] for i in range(start, end)]
            c, d, lower_q, upper_q = [], [], [], []
            for value, scale in zip(params[start:end], scales[start:end]):
                lo = max(self.qmin, min(self.qmax, math.floor(float(value) / float(scale))))
                hi = max(self.qmin, min(self.qmax, math.ceil(float(value) / float(scale))))
                lower_q.append(int(lo))
                upper_q.append(int(hi))
                c.append(lo * scale - float(value))
                d.append((hi - lo) * scale)

            n = end - start
            q = [[0.0] * n for _ in range(n)]
            for i in range(n):
                q[i][i] += 2.0 * d[i] * sum(sub_h[i][j] * c[j] for j in range(n))
            for i in range(n):
                for j in range(n):
                    q[i][j] += d[i] * sub_h[i][j] * d[j]
            problems.append(_Problem(f"{prefix}.chunk{start // self.qubo_size}", q, list(keys[start:end]), lower_q, upper_q))
        return problems

    def _activation_hessian(self, layer: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Return the activation Gram matrix used by output-error PTQ."""

        x = inputs.detach().cpu()
        if isinstance(layer, nn.Conv2d):
            patches = F.unfold(x, layer.kernel_size, dilation=layer.dilation, padding=layer.padding, stride=layer.stride)
            mean_patches = patches.mean(dim=0).transpose(0, 1)
            return mean_patches.transpose(0, 1).matmul(mean_patches)
        mean_x = x.reshape(-1, layer.weight.shape[1]).mean(dim=0)
        return torch.outer(mean_x, mean_x)

    def _weight_key(self, shape: torch.Size, out_i: int, in_i: int) -> Tuple[Any, ...]:
        """Map one output-row offset to a tensor weight index."""

        if len(shape) == 2:
            return ("weight", out_i, in_i)
        kernel_size = int(shape[2] * shape[3])
        in_channel, kernel_offset = in_i // kernel_size, in_i % kernel_size
        return ("weight", out_i, in_channel, kernel_offset // int(shape[3]), kernel_offset % int(shape[3]))

    def _flat_weight_key(self, shape: torch.Size, index: int) -> Tuple[Any, ...]:
        """Map one flattened offset to a tensor weight index."""

        if len(shape) == 2:
            return ("weight", index // int(shape[1]), index % int(shape[1]))
        per_output = int(shape[1] * shape[2] * shape[3])
        return self._weight_key(shape, index // per_output, index % per_output)

    def _solve(self, problem: _Problem) -> Tuple[List[int], float, str]:
        """Solve one QUBO with configured solvers and keep the best sample."""

        best: Optional[Tuple[List[int], float, str]] = None
        for spec in self.solvers:
            name = str(spec.get("name", spec.get("solver", spec.get("type", "sa")))).lower()
            params = dict(spec.get("params", spec.get("parameters", {})))
            is_cim = "cim" in name.replace("_", "").replace(".", "")
            bits = _solve_cim(problem, params) if is_cim else _solve_sa(problem, params)
            value = _qubo_energy(problem.q, bits)
            solver_name = "kw.cim.CIMOptimizer" if is_cim else "kw.classical.SimulatedAnnealingOptimizer"
            if best is None or value < best[1]:
                best = (bits, value, solver_name)
        return best

    def _capture_inputs(self, data: Any) -> Dict[str, torch.Tensor]:
        """Collect one calibration input tensor for every quantizable layer."""

        captured: Dict[str, torch.Tensor] = {}
        hooks = []

        def save_input(layer_name: str):
            """Create a hook that stores a layer input tensor."""

            def hook(_module, inputs, _output):
                """Record the first input tensor seen by the hook."""

                captured.setdefault(layer_name, inputs[0].detach().cpu())

            return hook

        for name, layer in self._quantizable_layers():
            hooks.append(layer.register_forward_hook(save_input(name)))

        batch = data
        if not hasattr(batch, "detach") and not (isinstance(batch, (list, tuple)) and batch and hasattr(batch[0], "detach")):
            batch = next(iter(data))
        model_input = batch[0] if isinstance(batch, (list, tuple)) else batch
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.model(model_input.to(device))
        if was_training:
            self.model.train()
        for hook in hooks:
            hook.remove()
        return captured

    def _quantizable_layers(self):
        """Yield selected Conv2d and Linear layers from the working model."""

        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)) and self._layer_selected(name):
                yield name, layer

    def _layer_selected(self, name: str) -> bool:
        """Return whether a layer name matches the requested targets."""

        return not self.layers or any(fnmatch.fnmatchcase(name, pattern) for pattern in self.layers)

    def _nearest_q(self, value: float, scale: float) -> int:
        """Round one value into the configured integer range."""

        return max(self.qmin, min(self.qmax, round(float(value) / float(scale))))

    def _nearest_weight(self, values, scale: float):
        """Round a nested weight tensor into integer values."""

        if isinstance(values, list):
            return [self._nearest_weight(value, scale) for value in values]
        return self._nearest_q(float(values), scale)

    def _write_weight(self, layer: nn.Module, q_weight, scale: float) -> None:
        """Write dequantized integer weights back to one layer."""

        weight = torch.tensor(q_weight, dtype=layer.weight.dtype, device=layer.weight.device) * scale
        layer.weight.data.copy_(weight)


def _solve_sa(problem: _Problem, params: Dict[str, Any]) -> List[int]:
    """Solve one QUBO with Kaiwu simulated annealing."""

    defaults = {
        "initial_temperature": 100,
        "alpha": 0.99,
        "cutoff_temperature": 0.001,
        "iterations_per_t": 100,
        "size_limit": 1,
    }
    defaults.update(params)
    model = kw.qubo.qubo_matrix_to_qubo_model(np.array(problem.q, dtype=float))
    result, _value = kw.solver.SimpleSolver(kw.classical.SimulatedAnnealingOptimizer(**defaults)).solve_qubo(model)
    bits = [0] * len(problem.q)
    for key, value in result.items():
        match = re.search(r"\[(\d+)\]", str(key))
        if match:
            bits[int(match.group(1))] = int(round(float(value)))
    return bits


def _solve_cim(problem: _Problem, params: Dict[str, Any]) -> List[int]:
    """Solve one QUBO through Kaiwu CIMOptimizer after Ising conversion."""

    base_task = params.pop("task_name", "qubo_ptq")
    bit_width = params.pop("bit_width", 14)
    save_dir = Path(params.pop("save_dir", Path.cwd() / "kaiwu_cim_results"))
    save_dir.mkdir(parents=True, exist_ok=True)
    kw.common.CheckpointManager.save_dir = str(save_dir.resolve())
    params["task_name"] = _safe_name(f"{base_task}_{problem.name}")
    ising, _bias = kw.conversion.qubo_matrix_to_ising_matrix(np.array(problem.q, dtype=float))
    raw_size = ising.shape[0]
    if bit_width is not None:
        ising = kw.ising.adjust_ising_matrix_precision(ising, bit_width=bit_width)
    samples = kw.cim.CIMOptimizer(**params).solve(ising)
    rows = samples.tolist() if hasattr(samples, "tolist") else list(samples)
    if rows and not isinstance(rows[0], list):
        rows = [rows]

    best_bits, best_value = None, math.inf
    for row in rows:
        if raw_size == len(problem.q) + 1:
            aux = int(row[len(problem.q)])
            candidates = [
                [int((1 - int(row[i]) * aux) // 2) for i in range(len(problem.q))],
                [int((1 + int(row[i]) * aux) // 2) for i in range(len(problem.q))],
            ]
        else:
            candidates = [
                [int((1 + int(row[i])) // 2) for i in range(len(problem.q))],
                [int((1 - int(row[i])) // 2) for i in range(len(problem.q))],
            ]
        for bits in candidates:
            value = _qubo_energy(problem.q, bits)
            if value < best_value:
                best_bits, best_value = bits, value
    return best_bits or [0] * len(problem.q)


def _qubo_energy(qubo: Sequence[Sequence[float]], bits: Sequence[int]) -> float:
    """Evaluate one QUBO objective value."""

    return sum(qubo[i][j] for i, bi in enumerate(bits) if bi for j, bj in enumerate(bits) if bj)


def _scale(values: Sequence[float], qmax: int) -> float:
    """Compute a symmetric quantization scale."""

    max_abs = max([abs(float(value)) for value in values] or [0.0])
    return 1.0 if max_abs == 0.0 else max_abs / float(qmax)


def _safe_name(name: str) -> str:
    """Sanitize text for a Kaiwu task name."""

    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("._-")[:80] or "qubo_ptq"
