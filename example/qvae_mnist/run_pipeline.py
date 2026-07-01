#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line entry point for training the QVAE-MLP pipeline.
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Config
from downstream import get_full_pipeline
from utils.loadMNIST import loadMNIST


def flatten_dataloader(dataloader):
    """Convert a DataLoader to flattened numpy arrays."""
    X_list, y_list = [], []
    for data, labels in dataloader:
        X_list.append(data.view(data.size(0), -1).numpy())
        y_list.append(labels.numpy())
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train QVAE + MLP pipeline")
    # Model hyperparameters
    parser.add_argument("--model-type", type=str, default="MNISTQVAE", choices=["MNISTQVAE"])
    parser.add_argument("--num-latent-units", type=int, default=256)
    parser.add_argument("--dist-beta", type=float, default=10.0)
    parser.add_argument("--kl-beta", type=float, default=1e-6)
    parser.add_argument("--sampler-type", type=str, default="sa", choices=["sa", "cim"])
    parser.add_argument("--loss-type", type=str, default="bernoulli", choices=["bernoulli", "mse"])
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Data parameters
    parser.add_argument("--name", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--num-train-samples", type=int, default=60000)
    parser.add_argument("--num-test-samples", type=int, default=10000)

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--bm-lr", type=float, default=8e-4)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--feature-type", type=str, default="q", choices=["q", "zeta"])
    parser.add_argument("--run-tsne", action="store_true")
    parser.add_argument("--compute_energy", action="store_true")

    # Output
    parser.add_argument("--output-dir", type=str, default=None)

    # MLP classifier arguments
    parser.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--mlp-output-dim", type=int, default=10)
    parser.add_argument("--mlp-lr", type=float, default=8e-5)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)

    args = parser.parse_args()

    # Build Config object
    config = Config(
        model_type=args.model_type,
        name=args.name,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        bm_lr=args.bm_lr,
        use_cuda=args.use_cuda,
        feature_type=args.feature_type,
        run_tsne=args.run_tsne,
        compute_energy=args.compute_energy,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        output_dir=args.output_dir,
        classifier_kwargs={
            "hidden_dims": args.mlp_hidden_dims,
            "output_dim": args.mlp_output_dim,
            "lr_mlp": args.mlp_lr,
            "batch_size_mlp": args.mlp_batch_size,
            "epochs_mlp": args.mlp_epochs,
            "weight_decay": args.mlp_weight_decay,
        },
    )

    # Override model-specific hyperparameters (Config defaults may differ)
    config.dist_beta = args.dist_beta
    config.kl_beta = args.kl_beta
    config.sampler_type = args.sampler_type
    config.loss_type = args.loss_type
    config.weight_decay = args.weight_decay
    config.num_latent_units = args.num_latent_units

    # Load data (flattened)
    train_loader, test_loader = loadMNIST(
        name=config.name,
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_evts_train=config.num_train_samples,
        num_evts_test=config.num_test_samples,
        use_cuda=config.use_cuda,
    )
    X_train_raw, y_train_raw = flatten_dataloader(train_loader)
    X_test_raw, y_test_raw = flatten_dataloader(test_loader)

    # Build and train pipeline
    print("=== Building pipeline ===")
    pipeline = get_full_pipeline(config)
    pipeline.fit(X_train_raw, y_train_raw)

    # Evaluate
    accuracy = pipeline.score(X_test_raw, y_test_raw)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Optional: save results
    if config.output_dir:
        results = {"test_accuracy": accuracy}
        with open(os.path.join(config.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
