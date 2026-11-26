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
import math
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_timing import SamplerTimer, select_device, synchronize

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import EPConv2d, EPLinear, EquilibriumPropagation


def _load_splits(data_dir, n_val, seed, include_test=True):
    """Return MNIST train, validation and test arrays.

    The official 10000 test images are the test split. ``n_val`` images drawn
    at random from the 60000 training images are held out for validation.
    """
    train = MNIST(data_dir, train=True, download=True)
    if not 0 < n_val < len(train):
        raise ValueError("validation size must be smaller than the training set")
    features = train.data.reshape(len(train.data), -1).numpy().astype(np.float32) / 255.0
    labels = train.targets.numpy()
    order = np.random.default_rng(seed).permutation(len(features))
    val, fit = order[:n_val], order[n_val:]
    test_split = None
    if include_test:
        test = MNIST(data_dir, train=False, download=True)
        test_x = test.data.reshape(len(test.data), -1).numpy().astype(np.float32) / 255.0
        test_split = (test_x, test.targets.numpy())
    return (features[fit], labels[fit]), (features[val], labels[val]), test_split


def _build_model(args):
    """Build the fully connected or convolutional EP network."""
    if args.conv:
        conv = EPConv2d((1, 28, 28), out_channels=args.channels, kernel_size=7, stride=3)
        layers = [conv, EPLinear(conv.out_features, 10)]
    else:
        layers = [EPLinear(784, args.hidden), EPLinear(args.hidden, 10)]
    model = EquilibriumPropagation(layers)
    with torch.no_grad():
        model.layers[0].weight.mul_(args.input_scale)
    return model.to(getattr(args, "device", "cpu"))


def _dataset(split, limit):
    """Convert a split to a dataset, optionally retaining its first rows."""
    features, labels = split
    return TensorDataset(torch.from_numpy(features[:limit]), torch.from_numpy(labels[:limit]))


def _spin_targets(labels):
    """Encode integer labels as +1 for their class and -1 elsewhere."""
    return 2.0 * torch.nn.functional.one_hot(labels, 10).float() - 1.0


def _readout(model, states):
    """Return class predictions from the free-phase output fields."""
    output = model.layers[-1]
    return (output.drive(states[-2]) + output.bias_vector()).argmax(dim=1)


def _sampler_config(iterations, seed):
    """Return the Kaiwu annealing settings for a run."""
    return {
        "initial_temperature": 10.0,
        "alpha": 0.9,
        "cutoff_temperature": 0.01,
        "iterations_per_t": iterations,
        "size_limit": 4,
        "rand_seed": seed,
        "process_num": 1,
    }


def _accuracy(model, settings, loader):
    """Predict with a fresh, fixed-seed sampler for each evaluation."""
    sampler = SamplerTimer(SimulatedAnnealingOptimizer(**settings))
    correct = 0
    processed, reported = 0, time.perf_counter()
    with torch.no_grad():
        for inputs, labels in loader:
            device = next(model.parameters()).device
            inputs, labels = inputs.to(device), labels.to(device)
            states = model.equilibrate(inputs, sampler)
            correct += int((_readout(model, states) == labels).sum())
            processed += len(inputs)
            if time.perf_counter() - reported >= 30:
                print(json.dumps({"phase": "accuracy", "processed": processed,
                                  "total": len(loader.dataset)}), flush=True)
                reported = time.perf_counter()
    model._sa_seconds = getattr(model, "_sa_seconds", 0.0) + sampler.seconds
    model._sa_calls = getattr(model, "_sa_calls", 0) + sampler.calls
    return correct / len(loader.dataset)


def _positive_int(value):
    """Parse a strictly positive integer."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _parse_args() -> argparse.Namespace:
    """Parse training or checkpoint evaluation arguments."""
    parser = argparse.ArgumentParser(description="Equilibrium propagation on MNIST")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--limit-train", type=_positive_int, help="first N training images")
    parser.add_argument("--limit-eval", type=_positive_int, help="first N val/test images")
    parser.add_argument("--n-val", type=_positive_int, default=5000)
    parser.add_argument("--epochs", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--hidden", type=_positive_int, default=128)
    parser.add_argument("--conv", action="store_true", help="strided EPConv2d first layer")
    parser.add_argument("--channels", type=_positive_int, default=4)
    parser.add_argument("--input-scale", type=float, default=2.0)
    parser.add_argument("--sa-iterations", type=_positive_int, default=50,
                        help="annealing iterations per temperature")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--validation-only", action="store_true", help="skip the MNIST test set")
    parser.add_argument("--evaluate", type=Path, help="evaluate a saved checkpoint")
    parser.add_argument("--output-dir", type=Path, help="new directory for this run")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N; SA remains on CPU")
    parser.add_argument("--threads", type=_positive_int, default=1)
    args = parser.parse_args()
    try:
        args.device = str(select_device(args.device))
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    for name in ("lr", "beta", "input_scale"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.seed < 0 or args.split_seed < 0:
        parser.error("seeds must be nonnegative")
    return args


def _write_json(path, payload):
    """Replace a JSON file only after its new contents are written."""
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def _train_epoch(model, sampler, optimizer, loader, beta):
    """Apply EP updates and return the sample-weighted mean objective."""
    total = 0.0
    processed, reported = 0, time.perf_counter()
    for inputs, labels in loader:
        device = next(model.parameters()).device
        inputs, labels = inputs.to(device), labels.to(device)
        free = model.equilibrate(inputs, sampler)
        nudged = model.equilibrate(inputs, sampler, _spin_targets(labels), beta)
        optimizer.zero_grad()
        loss = model.objective(inputs, free, nudged, beta)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite EP objective")
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(inputs)
        processed += len(inputs)
        if time.perf_counter() - reported >= 30:
            print(json.dumps({"phase": "train", "processed": processed,
                              "total": len(loader.dataset)}), flush=True)
            reported = time.perf_counter()
    return total / len(loader.dataset)


def _loaders(config, data_dir, include_test):
    """Build evaluation loaders with consistent split and subset limits."""
    train, val, test = _load_splits(
        data_dir, config.n_val, config.split_seed, include_test=include_test
    )
    train = _dataset(train, config.limit_train)
    val = _dataset(val, config.limit_eval)
    test_loader = None if test is None else DataLoader(
        _dataset(test, config.limit_eval), batch_size=config.batch_size
    )
    return (
        DataLoader(train, batch_size=config.batch_size),
        DataLoader(val, batch_size=config.batch_size),
        test_loader,
    )


def _metrics(model, settings, train, val):
    """Measure train and validation classification accuracy and error."""
    synchronize(model)
    model._sa_seconds, model._sa_calls = 0.0, 0
    started = time.perf_counter()
    train_accuracy = _accuracy(model, settings, train)
    val_accuracy = _accuracy(model, settings, val)
    synchronize(model)
    elapsed = time.perf_counter() - started
    return {
        "eval_sa_seconds": model._sa_seconds,
        "eval_non_sa_seconds": elapsed - model._sa_seconds,
        "eval_sa_calls": model._sa_calls,
        "train_accuracy": train_accuracy,
        "train_error": 1.0 - train_accuracy,
        "val_accuracy": val_accuracy,
        "val_error": 1.0 - val_accuracy,
        "eval_seconds": elapsed,
    }


def _evaluate(args, checkpoint, output_dir):
    """Restore the saved model and evaluate the saved data configuration."""
    config = argparse.Namespace(**checkpoint["config"])
    config.device = args.device
    model = _build_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    settings = checkpoint["eval_sampler"]
    train, val, test = _loaders(
        config, args.data_dir, not (args.validation_only or checkpoint["config"]["validation_only"])
    )
    result = {
        "checkpoint": str(args.evaluate),
        "epoch": checkpoint["epoch"],
        **_metrics(model, settings, train, val),
    }
    if test is not None:
        synchronize(model)
        model._sa_seconds, model._sa_calls = 0.0, 0
        started = time.perf_counter()
        result["test_accuracy"] = _accuracy(model, settings, test)
        synchronize(model)
        result["test_seconds"] = time.perf_counter() - started
        result["test_sa_seconds"] = model._sa_seconds
        result["test_sa_calls"] = model._sa_calls
        result["test_non_sa_seconds"] = result["test_seconds"] - model._sa_seconds
    _write_json(output_dir / "evaluation.json", result)
    print(json.dumps(result), flush=True)


def _train(args, output_dir):
    """Train the model, saving a checkpoint and metrics after each epoch."""
    train, val, test = _loaders(args, args.data_dir, not args.validation_only)
    torch.manual_seed(args.seed)
    model = _build_model(args)
    settings = _sampler_config(args.sa_iterations, args.seed)
    eval_settings = _sampler_config(args.sa_iterations, 0)
    sampler = SamplerTimer(SimulatedAnnealingOptimizer(**settings))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    order = np.arange(len(train.dataset))
    rng = np.random.default_rng(args.seed)
    batches = DataLoader(train.dataset, batch_size=args.batch_size, sampler=order)
    config = {key: value for key, value in vars(args).items()
              if key not in ("data_dir", "output_dir", "evaluate")}
    summary = {
        **config,
        "execution": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                      "model_device": str(next(model.parameters()).device),
                      "sampler_device": "cpu", "threads": torch.get_num_threads(),
                      "tf32": False},
        "layer_sizes": list(model.layer_sizes),
        "n_train": len(train.dataset),
        "n_val": len(val.dataset),
        "n_test": 0 if test is None else len(test.dataset),
        "train_sampler": settings,
        "eval_sampler": eval_settings,
        "history": [],
    }
    summary_path = output_dir / ("summary_conv.json" if args.conv else "summary.json")
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        synchronize(model)
        sa_before, calls_before = sampler.seconds, sampler.calls
        started = time.perf_counter()
        objective = _train_epoch(model, sampler, optimizer, batches, args.beta)
        synchronize(model)
        elapsed = time.perf_counter() - started
        sa_elapsed = sampler.seconds - sa_before
        row = {"train_sa_seconds": sa_elapsed, "train_non_sa_seconds": elapsed - sa_elapsed,
               "train_sa_calls": sampler.calls - calls_before, "epoch": epoch, "ep_objective": objective,
               "train_seconds": elapsed}
        row.update(_metrics(model, eval_settings, train, val))
        summary["history"].append(row)
        torch.save(
            {"config": config, "epoch": epoch, "state_dict": model.state_dict(),
             "train_sampler": settings, "eval_sampler": eval_settings},
            output_dir / "checkpoint.tmp",
        )
        (output_dir / "checkpoint.tmp").replace(output_dir / "checkpoint.pt")
        _write_json(summary_path, summary)
        print(json.dumps(row), flush=True)

    if test is not None:
        synchronize(model)
        model._sa_seconds, model._sa_calls = 0.0, 0
        started = time.perf_counter()
        summary["test_accuracy"] = _accuracy(model, eval_settings, test)
        synchronize(model)
        summary["test_seconds"] = time.perf_counter() - started
        summary["test_sa_seconds"] = model._sa_seconds
        summary["test_sa_calls"] = model._sa_calls
        summary["test_non_sa_seconds"] = summary["test_seconds"] - model._sa_seconds
        _write_json(summary_path, summary)
    print(f"wrote {summary_path}", flush=True)


def main() -> None:
    """Train or evaluate MNIST EP in a new output directory."""
    args = _parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = None
    if args.evaluate is not None:
        checkpoint = torch.load(args.evaluate, map_location="cpu", weights_only=True)
    if checkpoint is not None:
        default_name = "evaluation"
    else:
        default_name = "conv" if args.conv else "linear"
    output_dir = args.output_dir or Path(__file__).resolve().parent / "results" / default_name
    output_dir.mkdir(parents=True, exist_ok=False)
    if checkpoint is None:
        _train(args, output_dir)
    else:
        _evaluate(args, checkpoint, output_dir)


if __name__ == "__main__":
    main()
