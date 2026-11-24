"""Local BP comparison on the same MNIST split and layer widths as EP."""

import argparse
import json
import runpy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy(model, loader):
    """Evaluate top-1 accuracy."""
    with torch.no_grad():
        return sum(int((model(x).argmax(dim=1) == y).sum()) for x, y in loader) / len(loader.dataset)


def main():
    """Run the fixed three-epoch BP baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    entry = Path(__file__).resolve().parents[1] / "train_mnist.py"
    ep = runpy.run_path(str(entry))
    config = argparse.Namespace(n_val=5000, split_seed=0, limit_train=None,
                                limit_eval=None, batch_size=20)
    train, val, test = ep["_loaders"](config, args.data_dir, True)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    order = np.arange(len(train.dataset))
    rng = np.random.default_rng(0)
    batches = DataLoader(train.dataset, batch_size=20, sampler=order)
    summary = {
        "model": "784-128-10 ReLU MLP", "epochs": 3, "seed": 0, "split_seed": 0,
        "n_train": len(train.dataset), "n_val": len(val.dataset), "n_test": len(test.dataset),
        "batch_size": 20, "lr": 0.05, "objective": "cross entropy", "history": [],
    }
    for epoch in range(1, 4):
        rng.shuffle(order)
        started = time.perf_counter()
        loss_sum = 0.0
        for inputs, labels in batches:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(inputs)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        train_accuracy, val_accuracy = accuracy(model, train), accuracy(model, val)
        row = {"epoch": epoch, "train_seconds": train_seconds,
               "eval_seconds": time.perf_counter() - started,
               "train_loss": loss_sum / len(train.dataset),
               "train_accuracy": train_accuracy, "val_accuracy": val_accuracy,
               "train_error": 1.0 - train_accuracy, "val_error": 1.0 - val_accuracy}
        summary["history"].append(row)
        ep["_write_json"](args.output_dir / "summary.json", summary)
        print(json.dumps(row), flush=True)
    summary["test_accuracy"] = accuracy(model, test)
    ep["_write_json"](args.output_dir / "summary.json", summary)
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
