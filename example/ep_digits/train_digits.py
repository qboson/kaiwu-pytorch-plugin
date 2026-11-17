# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Digits classification with equilibrium propagation.

Pixels are scaled to ``[0, 1]`` by dividing by 16. The EP network is 64-32-10
with ``+-1`` spins, and both phases are relaxed by Kaiwu simulated annealing.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import EquilibriumPropagation


def _load_splits(seed):
    """Return scaled Digits train, validation and test splits.

    The test split is the stratified 25% of ``random_state=0``. A further
    stratified 20% of the remaining images is held out for validation.
    """
    features, labels = load_digits(return_X_y=True)
    features = features.astype(np.float32) / 16.0
    labels = labels.astype(np.int64)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.25, random_state=0, stratify=labels
    )
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=int(seed), stratify=train_y
    )
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def _spin_targets(labels, n_classes=10):
    """Encode integer labels as ``+1`` for the class and ``-1`` elsewhere."""
    coded = -np.ones((len(labels), n_classes), dtype=np.float32)
    coded[np.arange(len(labels)), labels] = 1.0
    return torch.from_numpy(coded)


def _accuracy(model, sampler, features, labels):
    """Classify by the output field of the free-phase hidden spins."""
    inputs = torch.from_numpy(features)
    states = model.equilibrate(inputs, sampler)
    with torch.no_grad():
        readout = model.layers[-1]
        scores = readout.drive(states[-2]) + readout.bias_vector()
    return float((scores.argmax(dim=1).numpy() == labels).mean())


def main():
    """Train EP on Digits and write a small JSON summary."""
    parser = argparse.ArgumentParser(description="Equilibrium propagation on sklearn Digits")
    parser.add_argument("--smoke", action="store_true", help="100 train / 100 eval images")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument(
        "--input-scale", type=float, default=4.0, help="multiplier on the first-layer init"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = _load_splits(args.seed)
    if args.smoke:
        train_x, train_y = train_x[:100], train_y[:100]
        val_x, val_y = val_x[:100], val_y[:100]
        test_x, test_y = test_x[:100], test_y[:100]

    torch.manual_seed(args.seed)
    model = EquilibriumPropagation(layer_sizes=(64, args.hidden, 10))
    with torch.no_grad():
        model.layers[0].weight.mul_(args.input_scale)
    sampler = SimulatedAnnealingOptimizer(
        initial_temperature=10.0,
        alpha=0.9,
        cutoff_temperature=0.01,
        iterations_per_t=10,
        size_limit=4,
        rand_seed=args.seed,
        process_num=1,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    targets = _spin_targets(train_y)
    order = np.arange(len(train_x))
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            inputs = torch.from_numpy(train_x[batch])
            free = model.equilibrate(inputs, sampler)
            nudged = model.equilibrate(inputs, sampler, targets[batch], args.beta)
            optimizer.zero_grad()
            loss = model.objective(inputs, free, nudged, args.beta)
            loss.backward()
            optimizer.step()
        val_accuracy = _accuracy(model, sampler, val_x, val_y)
        seconds = time.perf_counter() - started
        history.append({"epoch": epoch + 1, "val_accuracy": val_accuracy, "seconds": seconds})
        print(f"epoch {epoch + 1}: val_accuracy={val_accuracy:.4f} ({seconds:.1f}s)")

    test_accuracy = _accuracy(model, sampler, test_x, test_y)
    summary = {
        "smoke": bool(args.smoke),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta": args.beta,
        "hidden": args.hidden,
        "input_scale": args.input_scale,
        "seed": args.seed,
        "n_train": len(train_x),
        "n_val": len(val_x),
        "n_test": len(test_x),
        "history": history,
        "test_accuracy": test_accuracy,
    }
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"test_accuracy={test_accuracy:.4f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
