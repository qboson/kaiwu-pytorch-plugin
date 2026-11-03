# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Run the spike-sorting feature-extraction comparison.

Example
-------
python example/spike_sorting/run_experiment.py \
    --data-dir path/to/spike_data \
    --simulations 1 4 16 35 \
    --models ae vae qbmvae \
    --epochs 100 \
    --out example/spike_sorting/results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import evaluate_clustering  # noqa: E402
from models import (  # noqa: E402
    Autoencoder,
    GaussianVAE,
    SpikeQBMVAE,
    extract_features,
    train_autoencoder,
    train_gaussian_vae,
    train_qbm_vae,
)
from spike_data import load_realdata, load_simulation  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce autoencoder feature extraction for spike sorting."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="directory containing simulation_<n>.mat and ground_truth.mat",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        nargs="+",
        default=[1, 4, 16, 35],
        help="simulation numbers used by the paper",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ae", "vae", "qbmvae"],
        choices=["ae", "vae", "qbmvae"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="example/spike_sorting/results")
    parser.add_argument(
        "--zscore", action="store_true", help="z-score every waveform before training"
    )
    parser.add_argument(
        "--realdata-dir",
        default=None,
        help="optional TINS .ssd directory; adds the real recording as a dataset",
    )
    parser.add_argument(
        "--max-spikes",
        type=int,
        default=None,
        help="optionally subsample spikes for quick runs",
    )
    return parser.parse_args()


def qbm_config(latent_dim: int) -> SimpleNamespace:
    """Build the configuration consumed by :class:`SpikeQBMVAE`."""
    return SimpleNamespace(
        num_latent_units=latent_dim,
        loss_type="mse",
        dist_beta=10.0,
        kl_beta=1e-5,
        weight_decay=0.0,
        encoder_hidden_nodes=(70, 60, 50, 40, 30, 20),
        decoder_hidden_nodes=(20, 30, 40, 50, 60, 70),
        bm_type="rbm",
        sampler_type="sa",
    )


def run_one(
    model_name: str,
    spikes: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Train one model and evaluate its clustering quality."""
    tensor = torch.from_numpy(spikes)
    started = time.time()
    if model_name == "ae":
        model = Autoencoder(input_dim=spikes.shape[1], latent_dim=2)
        train_autoencoder(
            model,
            tensor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
        )
    elif model_name == "vae":
        model = GaussianVAE(input_dim=spikes.shape[1], latent_dim=args.latent_dim)
        train_gaussian_vae(
            model,
            tensor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
        )
    elif model_name == "qbmvae":
        model = SpikeQBMVAE(
            input_dimension=spikes.shape[1],
            activation_fct=torch.tanh,
            config=qbm_config(args.latent_dim),
        )
        train_qbm_vae(
            model,
            tensor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
        )
    else:
        raise ValueError(model_name)

    features = extract_features(model, tensor, model_name, device=device).numpy()
    return {
        "model": model_name,
        "features": features,
        "seconds": time.time() - started,
    }


def main() -> None:
    """Entry point: run every model on every requested dataset."""
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    rows = []

    datasets = [
        (
            str(simulation),
            load_simulation(
                simulation,
                args.data_dir,
                zscore=args.zscore,
                max_spikes=args.max_spikes,
                seed=args.seed,
            ),
        )
        for simulation in args.simulations
    ]
    if args.realdata_dir:
        datasets.append(
            (
                "real",
                load_realdata(
                    args.realdata_dir,
                    zscore=args.zscore,
                    max_spikes=args.max_spikes,
                    seed=args.seed,
                ),
            )
        )

    for name, data in datasets:
        print(
            f"dataset {name}: {data.spikes.shape[0]} spikes, "
            f"{len(np.unique(data.labels))} clusters"
        )
        for model_name in args.models:
            result = run_one(model_name, data.spikes, args, device)
            metrics = evaluate_clustering(result["features"], data.labels, seed=args.seed)
            row = {
                "simulation": name,
                "model": model_name,
                "n_spikes": int(data.spikes.shape[0]),
                **metrics,
                "seconds": round(result["seconds"], 2),
            }
            rows.append(row)
            print(
                f"  {model_name:7s} ARI={row['ari']:.4f} AMI={row['ami']:.4f} "
                f"VM={row['v_measure']:.4f} SS={row['silhouette']:.4f} "
                f"({row['seconds']}s)"
            )

    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.out, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print(f"\nresults written to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()

