# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""MNIST classification with equilibrium propagation.

Images come from ``torchvision.datasets.MNIST`` with pixels in ``[0, 1]``. The
default network is 784-128-10 with ``+-1`` spins. ``--conv`` replaces the
first layer with a strided :class:`EPConv2d`. Both phases are relaxed by Kaiwu
simulated annealing.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torchvision.datasets import MNIST

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import EPConv2d, EPLinear, EquilibriumPropagation


def _load_splits(data_dir, n_val, seed):
    """Return MNIST train, validation and test arrays.

    The official 10000 test images are the test split. ``n_val`` images drawn
    at random from the 60000 training images are held out for validation.
    """
    train = MNIST(data_dir, train=True, download=True)
    test = MNIST(data_dir, train=False, download=True)
    features = train.data.reshape(len(train.data), -1).numpy().astype(np.float32) / 255.0
    labels = train.targets.numpy()
    order = np.random.default_rng(seed).permutation(len(features))
    val, fit = order[:n_val], order[n_val:]
    test_x = test.data.reshape(len(test.data), -1).numpy().astype(np.float32) / 255.0
    test_y = test.targets.numpy()
    return (features[fit], labels[fit]), (features[val], labels[val]), (test_x, test_y)


def _build_model(args):
    """Build the MLP or the convolutional EP network."""
    if args.conv:
        conv = EPConv2d((1, 28, 28), out_channels=args.channels, kernel_size=7, stride=3)
        layers = [conv, EPLinear(conv.out_features, 10)]
    else:
        layers = [EPLinear(784, args.hidden), EPLinear(args.hidden, 10)]
    model = EquilibriumPropagation(layers)
    with torch.no_grad():
        model.layers[0].weight.mul_(args.input_scale)
    return model


def _spin_targets(labels, n_classes=10):
    """Encode integer labels as ``+1`` for the class and ``-1`` elsewhere."""
    coded = -np.ones((len(labels), n_classes), dtype=np.float32)
    coded[np.arange(len(labels)), labels] = 1.0
    return torch.from_numpy(coded)


def _accuracy(model, sampler, features, labels):
    """Classify by the output field of the free-phase hidden spins."""
    states = model.equilibrate(torch.from_numpy(features), sampler)
    with torch.no_grad():
        readout = model.layers[-1]
        scores = readout.drive(states[-2]) + readout.bias_vector()
    return float((scores.argmax(dim=1).numpy() == labels).mean())


def _parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Equilibrium propagation on MNIST")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--limit-train", type=int, default=None, help="first N training images")
    parser.add_argument("--limit-eval", type=int, default=None, help="first N val/test images")
    parser.add_argument("--n-val", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--conv", action="store_true", help="strided EPConv2d first layer")
    parser.add_argument("--channels", type=int, default=4, help="output channels with --conv")
    parser.add_argument(
        "--input-scale", type=float, default=2.0, help="multiplier on the first-layer init"
    )
    parser.add_argument(
        "--sa-iterations", type=int, default=50, help="annealing iterations per temperature"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    """Train EP on MNIST and write a small JSON summary."""
    args = _parse_args()
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = _load_splits(
        args.data_dir, args.n_val, args.seed
    )
    train_x, train_y = train_x[: args.limit_train], train_y[: args.limit_train]
    val_x, val_y = val_x[: args.limit_eval], val_y[: args.limit_eval]
    test_x, test_y = test_x[: args.limit_eval], test_y[: args.limit_eval]

    torch.manual_seed(args.seed)
    model = _build_model(args)
    sampler = SimulatedAnnealingOptimizer(
        initial_temperature=10.0,
        alpha=0.9,
        cutoff_temperature=0.01,
        iterations_per_t=args.sa_iterations,
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
        train_seconds = time.perf_counter() - started
        val_accuracy = _accuracy(model, sampler, val_x, val_y)
        history.append(
            {"epoch": epoch + 1, "val_accuracy": val_accuracy, "train_seconds": train_seconds}
        )
        print(
            f"epoch {epoch + 1}: val_accuracy={val_accuracy:.4f} "
            f"(train {train_seconds:.1f}s, {1000 * train_seconds / (2 * len(order)):.1f} ms/solve)",
            flush=True,
        )

    test_accuracy = _accuracy(model, sampler, test_x, test_y)
    summary = {
        "conv": args.conv,
        "layer_sizes": list(model.layer_sizes),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta": args.beta,
        "input_scale": args.input_scale,
        "sa_iterations": args.sa_iterations,
        "seed": args.seed,
        "n_train": len(train_x),
        "n_val": len(val_x),
        "n_test": len(test_x),
        "history": history,
        "test_accuracy": test_accuracy,
    }
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    name = "summary_conv.json" if args.conv else "summary.json"
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"test_accuracy={test_accuracy:.4f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
