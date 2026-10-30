"""Data, model and sampler helpers shared by the simple QDEQ ETTh1 examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import kaiwu as kw
from kaiwu.cim import CIMOptimizer
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import QDEQ, QDEQConfig
from kaiwu.torch_plugin._kaiwu_compat import PrecisionReducer

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = EXAMPLE_ROOT / "data" / "ETTh1.csv"
DEFAULT_CHECKPOINT = EXAMPLE_ROOT / "checkpoints" / "simple_etth1.pt"

# Standard ETTh1 split used by iTransformer: 12 / 4 / 4 months of hourly data.
TRAIN_ROWS, VAL_ROWS, TEST_ROWS = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24


# Data.


class ETTh1Windows(Dataset):
    """Sliding windows over one split of ETTh1, scaled with training statistics.

    Args:
        values: Scaled series of shape (time, variables).
        seq_len: Look-back length.
        pred_len: Forecast horizon.
        stride: Step between consecutive window starts.
    """

    def __init__(self, values: np.ndarray, seq_len: int, pred_len: int, stride: int = 1):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.starts = list(range(0, len(values) - seq_len - pred_len + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = self.starts[index]
        mid = start + self.seq_len
        return self.values[start:mid], self.values[mid : mid + self.pred_len]


def build_loaders(
    data_path: Path,
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    train_stride: int = 4,
    eval_stride: int = 8,
) -> dict[str, DataLoader]:
    """Builds train / val / test loaders for the standard ETTh1 split.

    Strides above 1 subsample the windows to keep the example small.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Download ETTh1.csv as described in example/qdeq/README.md."
        )
    values = pd.read_csv(data_path).drop(columns=["date"]).to_numpy(dtype=np.float32)
    train = values[:TRAIN_ROWS]
    mean, std = train.mean(axis=0), train.std(axis=0) + 1e-8
    scaled = (values - mean) / std

    # Validation and test windows start seq_len rows early so their first
    # forecast begins exactly at the split boundary.
    bounds = {
        "train": (0, TRAIN_ROWS),
        "val": (TRAIN_ROWS - seq_len, TRAIN_ROWS + VAL_ROWS),
        "test": (TRAIN_ROWS + VAL_ROWS - seq_len, TRAIN_ROWS + VAL_ROWS + TEST_ROWS),
    }
    loaders = {}
    for split, (begin, end) in bounds.items():
        stride = train_stride if split == "train" else eval_stride
        windows = ETTh1Windows(scaled[begin:end], seq_len, pred_len, stride)
        loaders[split] = DataLoader(windows, batch_size=batch_size, shuffle=split == "train")
    return loaders


# Model.


class EquilibriumBlock(nn.Module):
    """Weight-tied variate-attention block ``z_next = block(norm(z + x))``.

    Tokens are the embedded series of each variable, as in iTransformer.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mid = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Returns the next iterate, shaped like ``z``.

        Args:
            z: Current tokens of shape (batch, variables, d_model).
            x: Embedded input tokens injected at every call.

        Returns:
            torch.Tensor: Next tokens of shape (batch, variables, d_model).
        """
        h = self.norm_in(z + x)
        h = self.norm_mid(h + self.attention(h, h, h, need_weights=False)[0])
        return self.norm_out(h + self.ffn(h))


class AndersonDEQ(QDEQ):
    """Classical baseline: ``QDEQ`` with the forward fixed point solved by Anderson acceleration.

    Only ``solve_fixed_point`` changes; the implicit backward pass is inherited.
    Uses ``f_thres``, ``f_eps``, ``stop_mode``, ``anderson_m`` and
    ``anderson_lam`` from the config.

    Args:
        layer: Weight-tied layer ``layer(z, x) -> z_next``.
        config: Solver configuration.
    """

    def __init__(self, layer: nn.Module, config: QDEQConfig):
        super().__init__(layer, None, config)

    def solve_fixed_point(self, func, z0):
        """Finds ``z = func(z)`` by Anderson-accelerated iteration.

        Args:
            func: Fixed-point map; takes and returns tensors shaped like ``z0``.
            z0: Initial guess with the batch axis first.

        Returns:
            tuple: The iterate with the lowest tracked residual, and a dict with
            ``"nstep"`` and ``"residual"``.
        """
        cfg = self.config
        bsz, m = z0.shape[0], max(cfg.anderson_m, 2)
        x_hist = z0.new_zeros(bsz, m, z0[0].numel())
        f_hist = torch.zeros_like(x_hist)
        x_hist[:, 0] = z0.reshape(bsz, -1)
        f_hist[:, 0] = func(z0).reshape(bsz, -1)
        x_hist[:, 1] = f_hist[:, 0]
        f_hist[:, 1] = func(f_hist[:, 0].reshape_as(z0)).reshape(bsz, -1)
        # Bordered system whose solution gives mixing weights summing to one.
        h_mat = z0.new_zeros(bsz, m + 1, m + 1)
        h_mat[:, 0, 1:] = h_mat[:, 1:, 0] = 1
        rhs = z0.new_zeros(bsz, m + 1, 1)
        rhs[:, 0] = 1

        best_z, best_res, nstep = z0, float("inf"), 0
        for k in range(2, max(cfg.f_thres, 3)):
            n = min(k, m)
            g_mat = f_hist[:, :n] - x_hist[:, :n]
            eye = torch.eye(n, dtype=z0.dtype, device=z0.device)[None]
            h_mat[:, 1 : n + 1, 1 : n + 1] = g_mat @ g_mat.transpose(1, 2) + cfg.anderson_lam * eye
            alpha = torch.linalg.solve(h_mat[:, : n + 1, : n + 1], rhs[:, : n + 1])[:, 1 : n + 1, 0]
            x_hist[:, k % m] = (alpha[:, None] @ f_hist[:, :n])[:, 0]
            f_hist[:, k % m] = func(x_hist[:, k % m].reshape_as(z0)).reshape(bsz, -1)

            nstep = k - 1
            residual = (f_hist[:, k % m] - x_hist[:, k % m]).norm().item()
            if cfg.stop_mode == "rel":
                residual /= f_hist[:, k % m].norm().item() + 1e-9
            if residual < best_res:
                best_res, best_z = residual, x_hist[:, k % m].reshape_as(z0).clone()
            if residual < cfg.f_eps:
                break
        return best_z, {"nstep": nstep, "residual": best_res}


class DEQForecaster(nn.Module):
    """Small iTransformer-style forecaster whose encoder is a single ``QDEQ`` layer.

    Args:
        seq_len: Look-back length.
        pred_len: Forecast horizon.
        sampler: Kaiwu optimizer for the Ising forward solver, or ``None`` for
            the ``AndersonDEQ`` baseline.
        config: ``QDEQ`` solver configuration.
        d_model: Token width.
        n_heads: Attention heads.
        d_ff: Feed-forward width.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        sampler,
        config: QDEQConfig,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
    ):
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model)
        block = EquilibriumBlock(d_model, n_heads, d_ff)
        if sampler is None:
            self.deq = AndersonDEQ(block, config)
        else:
            self.deq = QDEQ(block, sampler, config)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Maps a (batch, seq_len, variables) history to a (batch, pred_len, variables) forecast."""
        # Normalise each window per variable, embed each variable's series as
        # one token, solve the equilibrium over tokens and project to the horizon.
        mean = history.mean(dim=1, keepdim=True)
        std = history.std(dim=1, keepdim=True) + 1e-5
        tokens = self.embed(((history - mean) / std).transpose(1, 2))
        forecast = self.head(self.deq(tokens)).transpose(1, 2)
        return forecast * std + mean


# Samplers.


def build_sampler(sampler_type: str, seed: int = 0):
    """Creates the optimizer used by the Ising forward solver.

    Args:
        sampler_type: ``"sa"`` for Kaiwu simulated annealing, ``"cim"`` for a
            Kaiwu CIM task (requires a licensed account), ``"anderson"`` for no sampler.
        seed: Random seed for simulated annealing.

    Returns:
        A Kaiwu optimizer, or ``None`` for ``"anderson"``.
    """
    if sampler_type == "anderson":
        return None
    if sampler_type == "sa":
        # A short annealing schedule keeps the example fast; QDEQ decodes
        # every returned row and keeps the best one.
        return SimulatedAnnealingOptimizer(
            initial_temperature=100.0,
            alpha=0.95,
            cutoff_temperature=0.1,
            iterations_per_t=5,
            size_limit=20,
            rand_seed=seed,
        )
    if sampler_type == "cim":
        # PrecisionReducer rounds the Ising matrix to the precision the CIM accepts.
        kw.common.CheckpointManager.save_dir = str(EXAMPLE_ROOT / "tmp")
        return PrecisionReducer(CIMOptimizer(task_name="qdeq_etth1", wait=True), precision=8)
    raise ValueError(f"Unsupported sampler type: {sampler_type}")


def build_config() -> QDEQConfig:
    """Solver settings sized for this example, shared by all solvers."""
    return QDEQConfig(
        f_thres=12,
        b_thres=15,
        num_basis=4,
        coeff_bits=4,
        coeff_range=2.0,
        fd_eps=1e-2,
    )


@torch.no_grad()
def evaluate(model: DEQForecaster, loader: DataLoader) -> dict[str, float]:
    """Returns scaled-space MSE and MAE plus the mean forward solver steps."""
    model.eval()
    squared, absolute, count, steps = 0.0, 0.0, 0, []
    for history, target in loader:
        forecast = model(history)
        squared += torch.sum((forecast - target) ** 2).item()
        absolute += torch.sum(torch.abs(forecast - target)).item()
        count += target.numel()
        steps.append(model.deq.get_last_stats()["forward_nstep"])
    if count == 0:
        raise ValueError("evaluate() needs a loader with at least one sample")
    return {"mse": squared / count, "mae": absolute / count, "steps": float(np.mean(steps))}
