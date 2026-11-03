# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Two-phase equilibrium-propagation updates and free-phase prediction."""

from __future__ import annotations

import hashlib

import torch


def derived_seed(seed, epoch, index):
    """Derive the anneal seed shared by one sample's free and nudged phases.

    Args:
        seed (int): Experiment seed.
        epoch (int): Epoch index.
        index (int): Sample index within the experiment, not just the batch.

    Returns:
        int: An integer in ``[0, 2**31)``.
    """
    material = "|".join((str(int(seed)), "pair", str(int(epoch)), str(int(index))))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def train_batch(model, inputs, targets, optimizer, *, beta, backend="sa", epoch=0, offset=0):
    """Solve every sample in a batch, then take one minimizer step.

    The loss is ``(E_nudged - E_free) / beta``. States are detached, so the
    gradient is the free-minus-nudged correlation difference on every layer.

    Args:
        model (EquilibriumPropagation): Network being trained.
        inputs (torch.Tensor): Shape ``(batch, n_in)`` or ``(n_in,)``.
        targets (torch.Tensor): One-hot targets in ``[0, 1]``.
        optimizer (torch.optim.Optimizer): Consumes ``.grad`` with one ``step``.
        beta (float): Strictly positive nudging coefficient.
        backend (str): Backend name forwarded to ``equilibrate``.
        epoch (int): Epoch index mixed into the paired seed.
        offset (int): Sample index of ``inputs[0]`` in the epoch.

    Returns:
        dict: ``batch_size``, ``loss``, ``free_cost``, ``accuracy``,
        ``zero_update`` and ``solver_calls``.

    Raises:
        ValueError: If ``beta`` is not strictly positive.
        RuntimeError: If a parameter changes while the batch is being solved.
    """
    if float(beta) <= 0.0:
        raise ValueError("train_batch requires a strictly positive beta")
    features, labels = _as_batch(model, inputs, targets)
    snapshot = _snapshot(model)
    losses, costs, correct, unchanged, calls = _collect_pairs(
        model, features, labels, float(beta), backend, int(epoch), int(offset), snapshot
    )
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    count = int(features.shape[0])
    return {
        "batch_size": count,
        "loss": float(loss.detach()),
        "free_cost": sum(costs) / count,
        "accuracy": correct / count,
        "zero_update": unchanged / count,
        "solver_calls": calls,
    }


def predict(model, inputs, backend="sa", seed=None):
    """Return free-phase class indices. Parameters are left unchanged.

    Args:
        model (EquilibriumPropagation): Network used for the free phase.
        inputs (torch.Tensor): Shape ``(batch, n_in)`` or ``(n_in,)``.
        backend (str): Backend name forwarded to ``equilibrate``.
        seed (int, optional): Base seed. Defaults to ``model.config.seed``.

    Returns:
        torch.Tensor: Shape ``(batch,)`` and dtype ``torch.long``. Ties keep
        the smallest index.
    """
    blank = torch.zeros(model.layer_sizes[-1], dtype=model.dtype)
    features, _labels = _as_batch(model, inputs, blank)
    snapshot = _snapshot(model)
    base = model.config.seed if seed is None else int(seed)
    classes = []
    for index in range(int(features.shape[0])):
        result = model.equilibrate(
            features[index],
            beta=0.0,
            backend=backend,
            seed=derived_seed(base, 0, index),
        )
        _require_snapshot(model, snapshot)
        classes.append(_argmax_stable(result["states"][-1]))
    return torch.tensor(classes, dtype=torch.long)


def _collect_pairs(model, features, labels, beta, backend, epoch, offset, snapshot):
    """Solve the paired phases for every row without touching parameters."""
    losses = []
    costs = []
    correct = 0
    unchanged = 0
    for index in range(int(features.shape[0])):
        sample = features[index]
        target = labels[index]
        pair_seed = derived_seed(model.config.seed, epoch, offset + index)
        free = model.equilibrate(sample, target, beta=0.0, backend=backend, seed=pair_seed)
        _require_snapshot(model, snapshot)
        nudged = model.equilibrate(
            sample,
            target,
            beta=beta,
            backend=backend,
            seed=pair_seed,
            init_solution=free["init_solution"],
        )
        _require_snapshot(model, snapshot)
        free_states = [state.detach() for state in free["states"]]
        nudged_states = [state.detach() for state in nudged["states"]]
        losses.append(
            (model.energy(sample, nudged_states) - model.energy(sample, free_states)) / beta
        )
        costs.append(float(model.cost(free_states[-1], target).detach()))
        correct += int(_argmax_stable(free_states[-1]) == int(torch.argmax(target).item()))
        if all(torch.equal(left, right) for left, right in zip(free_states, nudged_states)):
            unchanged += 1
    return losses, costs, correct, unchanged, 2 * int(features.shape[0])


def _as_batch(model, inputs, targets):
    """Promote one sample to a batch on the model's dtype and device."""
    device = model.biases[0].device
    features = torch.as_tensor(inputs, dtype=model.dtype, device=device)
    labels = torch.as_tensor(targets, dtype=model.dtype, device=device)
    if features.ndim == 1:
        features = features.unsqueeze(0)
        labels = labels.unsqueeze(0)
    return features, labels


def _snapshot(model):
    """Copy every parameter tensor."""
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def _require_snapshot(model, snapshot):
    """Raise when a solve mutated a parameter."""
    for name, value in model.named_parameters():
        if not torch.equal(value, snapshot[name]):
            raise RuntimeError(f"parameter {name} changed while solving a phase pair")


def _argmax_stable(state):
    """Return the smallest index among near-maximum entries."""
    values = state.detach()
    best = values.max()
    tied = torch.nonzero(values >= best - 1e-12, as_tuple=False)
    return int(tied[0].item())
