# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""Unsupervised GBRBM training loop with device and random seed management.

Only normal training rows reach this module: no label, validation or test value
is used to build the contrastive-divergence gradient.
"""

import random

import numpy as np
import torch
from ad_config import ModelConfig, TrainConfig

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def resolve_device(device_spec: str) -> torch.device:
    """Resolve the requested device specification.

    Args:
        device_spec (str): ``"auto"``, ``"cpu"`` or ``"cuda"``.

    Returns:
        torch.device: The resolved device.

    Raises:
        ValueError: If ``"cuda"`` is requested without an available GPU, or if the
            specification is unknown.
    """
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_spec == "cpu":
        return torch.device("cpu")
    if device_spec == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' was requested but CUDA is not available")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device specification: {device_spec!r}")


def set_global_seed(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch random number generators.

    Args:
        seed (int): Seed shared by the data split, the model init and the sampler.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(
    num_visible: int,
    model_config: ModelConfig,
    device: torch.device,
) -> GaussianBernoulliRestrictedBoltzmannMachine:
    """Construct the GBRBM and freeze ``log_var`` when requested."""
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=num_visible,
        num_hidden=model_config.num_hidden,
        is_visible_gaussian=True,
        eps=model_config.eps,
        dtype=torch.float32,
        device=device,
    )
    if model_config.freeze_log_var:
        model.log_var.requires_grad_(False)
    return model


def _train_step(
    model: GaussianBernoulliRestrictedBoltzmannMachine,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    sampler,
) -> float:
    """Run one contrastive-divergence step and return the scalar objective."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer.zero_grad()
    s_positive = model.infer_from_gaussian(batch, binarize=False)
    s_negative = model.sample(sampler)
    regularizer = train_config.weight_decay * (
        model.quadratic_coef.square().sum() + model.linear_bias.square().sum()
    )
    loss = model.objective(s_positive, s_negative) + regularizer
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, max_norm=train_config.grad_clip_norm)
    optimizer.step()
    return float(loss.detach())


def _run_epoch(
    model: GaussianBernoulliRestrictedBoltzmannMachine,
    dataset: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    sampler,
    generator: torch.Generator,
) -> float:
    """Run one epoch over shuffled mini-batches and return the mean objective."""
    num_samples = dataset.shape[0]
    order = torch.randperm(num_samples, generator=generator)
    total = 0.0
    counted = 0
    for start in range(0, num_samples, train_config.batch_size):
        index = order[start : start + train_config.batch_size]
        total += _train_step(model, dataset[index], optimizer, train_config, sampler) * len(index)
        counted += len(index)
    return total / counted


def _mean_free_energy(
    model: GaussianBernoulliRestrictedBoltzmannMachine,
    dataset: torch.Tensor,
) -> float:
    """Return the mean free energy of the normal training set."""
    with torch.no_grad():
        return float(model.marginal_energy(dataset).mean())


def train_gbrbm(
    x_train: np.ndarray,
    model_config: ModelConfig,
    train_config: TrainConfig,
    sampler,
) -> tuple[GaussianBernoulliRestrictedBoltzmannMachine, list[dict[str, float]]]:
    """Train a GBRBM on normal-only data and return the model and its history.

    Args:
        x_train (np.ndarray): Standardized normal training rows, shape ``(N, D)``.
        model_config (ModelConfig): Architecture and numerical settings.
        train_config (TrainConfig): Optimization hyper-parameters.
        sampler: Kaiwu Ising sampler used for the negative phase.

    Returns:
        tuple: The trained model and a per-epoch history of
        ``{"epoch", "mean_objective", "train_free_energy"}`` entries.

    Raises:
        ValueError: If ``x_train`` is not a non-empty finite 2D array.
    """
    data = np.ascontiguousarray(np.asarray(x_train, dtype=np.float32))
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("x_train must be a non-empty 2D array of shape (n_samples, n_features)")
    if not np.isfinite(data).all():
        raise ValueError("x_train must contain only finite values")

    device = resolve_device(train_config.device)
    set_global_seed(train_config.seed)
    model = _build_model(data.shape[1], model_config, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        trainable,
        lr=train_config.learning_rate,
        momentum=train_config.momentum,
    )
    dataset = torch.from_numpy(data).to(device=device, dtype=model.dtype)
    generator = torch.Generator().manual_seed(train_config.seed)

    history: list[dict[str, float]] = []
    for epoch in range(train_config.epochs):
        objective = _run_epoch(model, dataset, optimizer, train_config, sampler, generator)
        history.append(
            {
                "epoch": float(epoch + 1),
                "mean_objective": objective,
                "train_free_energy": _mean_free_energy(model, dataset),
            }
        )
    return model, history
