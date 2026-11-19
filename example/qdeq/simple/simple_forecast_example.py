"""Minimal forecasting example for ``QDEQ`` on ETTh1.

Loads the checkpoint written by ``simple_train_example.py`` and evaluates it on
the test split with the chosen forward solver, so the same weights can be
compared under Kaiwu SA, CIM and Anderson.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--original-units", action="store_true",
                        help="Also report MAE and MSE in each CSV column's original units")
    return parser.parse_args()


def main() -> None:
    """Runs the example."""
    args = parse_args()
    loaders = build_loaders(args.data_path)
    sampler = build_sampler(args.sampler, args.seed)
    model = DEQForecaster(96, 96, sampler, build_config())
    model.load_state_dict(torch.load(args.checkpoint)["state_dict"])

    test = evaluate(model, loaders["test"], original_units=args.original_units)
    print(
        f"{args.sampler}: test mse {test['mse']:.4f} mae {test['mae']:.4f} "
        f"(mean forward steps {test['steps']:.1f})"
    )
    if args.original_units:
        for name, errors in test["original_units"].items():
            print(f"  {name}: mse {errors['mse']:.4f} mae {errors['mae']:.4f} (original units)")
    history, target = next(iter(loaders["test"]))
    with torch.no_grad():
        forecast = model(history[:1])
    print("first test window, variable OT, next 8 hours (scaled):")
    print("  forecast", [round(v, 3) for v in forecast[0, :8, -1].tolist()])
    print("  actual  ", [round(v, 3) for v in target[0, :8, -1].tolist()])


if __name__ == "__main__":
    main()
