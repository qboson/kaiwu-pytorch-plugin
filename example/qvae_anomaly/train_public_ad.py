# -*- coding: utf-8 -*-
"""Train CleanQVAE on public AD benchmarks (creditcard, thyroid).

Usage:
    python train_public_ad.py --dataset thyroid --epochs 20
    python train_public_ad.py --dataset creditcard --epochs 20
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import Config, build_model  # noqa: E402
from trainer.trainer import train_model, set_seed  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.datasets import load as load_dataset  # noqa: E402

# architecture defaults per dataset
ARCH = {
    "creditcard": {"hidden": 128, "latent": 32},
    "thyroid":    {"hidden": 32,  "latent": 16},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(ARCH))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--kl-beta", type=float, default=1e-4)
    p.add_argument("--lambda-anom", type=float, default=50.0)
    p.add_argument("--out-dir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"train_{args.dataset}", log_file=str(out / "train.log"))
    logger.info("===== Training cleanqvae on %s =====", args.dataset)

    d = load_dataset(args.dataset)
    xtr = d["x_train"].numpy(); ytr = d["y_train"].numpy().astype(np.int64)
    xval = d["x_val"].numpy(); yval = d["y_val"].numpy().astype(np.int64)
    dim = int(xtr.shape[1])
    arch = ARCH[args.dataset]
    logger.info("data dim=%d train=%s val=%s", dim, xtr.shape, xval.shape)

    cfg = Config(
        input_dimension=dim, hidden_dim=arch["hidden"],
        num_var1=arch["latent"], num_var2=arch["latent"],
        lambda_anom=args.lambda_anom,
        kl_beta=args.kl_beta, lr=args.lr, batch_size=args.batch_size,
        epochs=args.epochs, seed=args.seed, out_dir=str(out),
    )
    set_seed(cfg.seed)

    model = build_model(cfg)
    logger.info("Model: %s", type(model).__name__)

    model, best = train_model(model, cfg, xtr, ytr, xval, yval,
                              tag=f"qvae_{args.dataset}", log_file=str(out / "train.log"))
    logger.info("Best epoch=%d valPR=%.4f", best["epoch"], best["val_pr"])

    # Save checkpoint
    ckpt_path = out / f"{args.dataset}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(cfg),
        "best_epoch": best["epoch"],
        "best_val_pr": best["val_pr"],
        "scaler": d.get("scaler"),
    }, ckpt_path)
    logger.info("Saved checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    main()
