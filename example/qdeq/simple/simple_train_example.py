"""Minimal training example for ``QDEQ`` on ETTh1 (96-step look-back, 96-step horizon).

The script keeps only the smallest useful training path:

1. build ETTh1 loaders on subsampled windows
2. build a small forecaster whose encoder is one ``QDEQ`` layer
3. train with the Kaiwu SA forward solver (or Anderson / CIM)
4. report validation and test error and save a checkpoint
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from etth1_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_PATH,
    DEQForecaster,
    build_config,
    build_loaders,
    build_sampler,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    """Parses command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sampler", choices=["sa", "anderson", "cim"], default="sa")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Runs the example."""
    args = parse_args()
    torch.manual_seed(args.seed)
    loaders = build_loaders(args.data_path)
    sampler = build_sampler(args.sampler, args.seed)
    model = DEQForecaster(96, 96, sampler, build_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, losses = time.perf_counter(), []
        for batch, (history, target) in enumerate(loaders["train"]):
            if batch >= args.max_train_batches:
                break
            loss = torch.nn.functional.mse_loss(model(history), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        stats = model.deq.get_last_stats()
        val = evaluate(model, loaders["val"])
        print(
            f"epoch {epoch}: train mse {sum(losses) / len(losses):.4f} | val mse {val['mse']:.4f} "
            f"mae {val['mae']:.4f} | forward steps {stats['forward_nstep']} "
            f"backward steps {stats['backward_nstep']} | {time.perf_counter() - start:.0f}s"
        )

    test = evaluate(model, loaders["test"])
    print(
        f"test mse {test['mse']:.4f} mae {test['mae']:.4f} "
        f"(mean forward steps {test['steps']:.1f})"
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "sampler": args.sampler}, args.checkpoint)
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
