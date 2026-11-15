# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Digits classification with equilibrium propagation and a backprop baseline.

Pixels are scaled to ``[0, 1]`` by dividing by 16. The EP network is 64-32-10
with one bit per unit, relaxed by Kaiwu simulated annealing. The baseline is an
MLP of the same widths trained with cross-entropy.
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

from kaiwu.torch_plugin import EPConfig, EquilibriumPropagation


def _load_splits(n_train, n_test, seed):
    """Return scaled Digits arrays and 10-way one-hot targets."""
    digits = load_digits()
    features = digits.data.astype(np.float64) / 16.0
    labels = digits.target.astype(np.int64)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.25, random_state=0, stratify=labels
    )
    if n_train is not None:
        train_x, train_y = train_x[: int(n_train)], train_y[: int(n_train)]
    if n_test is not None:
        test_x, test_y = test_x[: int(n_test)], test_y[: int(n_test)]
    return train_x, test_x, train_y, test_y


def _one_hot(labels, n_classes=10):
    """Encode integer labels as float64 one-hot rows."""
    coded = np.zeros((len(labels), n_classes), dtype=np.float64)
    coded[np.arange(len(labels)), labels] = 1.0
    return coded


def _train_ep(train_x, train_y, test_x, test_y, epochs, batch_size, lr, seed):
    """Train the EP network and return its test accuracy.

    One bit per unit turns on only when its field exceeds ``1/2``. The first
    layer is therefore drawn at scale ``0.2`` so a Digits image activates some
    hidden units, and the readout starts near ``0.02`` so it does not drown the
    nudge. ``beta`` must be greater than 1: at ``beta == 1`` the nudged energy
    of the target bit ties the all-zero state, and the two phases stay equal.
    """
    config = EPConfig(layer_sizes=(64, 32, 10), state_bits=1, seed=int(seed))
    model = EquilibriumPropagation(config)
    with torch.no_grad():
        model.weights[0].mul_(20.0)
        for weight in list(model.weights)[1:]:
            weight.mul_(2.0)
        model.biases[-1].fill_(-0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr))
    targets = _one_hot(train_y)
    order = np.arange(len(train_x))
    for epoch in range(int(epochs)):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(order)
        for start in range(0, len(order), int(batch_size)):
            batch = order[start : start + int(batch_size)]
            model.train_batch(
                torch.tensor(train_x[batch]),
                torch.tensor(targets[batch]),
                optimizer,
                beta=2.0,
                backend="sa",
                epoch=epoch,
                offset=start,
            )
    predicted = model.predict(torch.tensor(test_x), backend="sa", seed=seed)
    return float((predicted.numpy() == test_y).mean())


def _train_bp(train_x, train_y, test_x, test_y, epochs, batch_size, lr, seed):
    """Train a same-shaped MLP and return its test accuracy."""
    torch.manual_seed(int(seed))
    network = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    optimizer = torch.optim.SGD(network.parameters(), lr=float(lr))
    loss_fn = nn.CrossEntropyLoss()
    features = torch.tensor(train_x, dtype=torch.float32)
    labels = torch.tensor(train_y, dtype=torch.long)
    order = np.arange(len(train_x))
    for epoch in range(int(epochs)):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(order)
        for start in range(0, len(order), int(batch_size)):
            batch = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss_fn(network(features[batch]), labels[batch]).backward()
            optimizer.step()
    with torch.no_grad():
        predicted = network(torch.tensor(test_x, dtype=torch.float32)).argmax(dim=1).numpy()
    return float((predicted == test_y).mean())


def main():
    """Run the EP and BP Digits comparison and write a small JSON summary."""
    parser = argparse.ArgumentParser(description="EP vs BP on sklearn Digits")
    parser.add_argument("--smoke", action="store_true", help="100/100 samples, 6 epochs")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_train = n_test = None
    epochs = args.epochs
    if args.smoke:
        n_train = n_test = 100
        epochs = 6
    train_x, test_x, train_y, test_y = _load_splits(n_train, n_test, args.seed)
    ep_accuracy = _train_ep(
        train_x, train_y, test_x, test_y, epochs, args.batch_size, args.lr, args.seed
    )
    bp_accuracy = _train_bp(
        train_x, train_y, test_x, test_y, epochs, args.batch_size, args.lr, args.seed
    )
    summary = {
        "smoke": bool(args.smoke),
        "epochs": int(epochs),
        "n_train": int(len(train_x)),
        "n_test": int(len(test_x)),
        "ep_accuracy": ep_accuracy,
        "bp_accuracy": bp_accuracy,
    }
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"ep_accuracy={ep_accuracy:.4f} bp_accuracy={bp_accuracy:.4f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
