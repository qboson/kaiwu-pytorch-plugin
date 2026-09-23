# -*- coding: utf-8 -*-
"""Evaluate CleanQVAE checkpoint on public AD benchmarks.

Usage:
    python evaluate_public_ad.py --dataset thyroid
    python evaluate_public_ad.py --dataset creditcard
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import Config, build_model  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.evaluate import optimize_threshold_rbm  # noqa: E402
from utils.visualize import (plot_score_distribution, plot_pr_roc_curve,
                              plot_latent_tsne, plot_timeseries_anomaly)  # noqa: E402
from utils.datasets import load as load_dataset  # noqa: E402

# which visualizations to run per dataset
VIZ_CONFIG = {
    "thyroid":    ["score_dist", "pr_roc"],
    "creditcard": ["score_dist", "pr_roc", "tsne", "timeseries"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(VIZ_CONFIG))
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"eval_{args.dataset}", log_file=str(out / "evaluate.log"))
    logger.info("===== Evaluating cleanqvae on %s =====", args.dataset)

    # Load checkpoint
    ckpt_path = args.ckpt or (out / f"{args.dataset}.pt")
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Best epoch=%d valPR=%.4f", ckpt["best_epoch"], ckpt["best_val_pr"])

    # Load data
    d = load_dataset(args.dataset)
    xval = d["x_val"].numpy(); yval = d["y_val"].numpy().astype(np.int64)
    xte = d["x_test"].numpy(); yte = d["y_test"].numpy().astype(np.int64)
    device = cfg.device

    # Optimize threshold on val
    def fe(xb):
        with torch.no_grad():
            return model.get_score_energy(xb.to(device)).cpu()

    val_loader = DataLoader(TensorDataset(torch.from_numpy(xval),
                                         torch.from_numpy(yval)),
                            batch_size=2048, shuffle=False)
    th = optimize_threshold_rbm(fe, val_loader, device,
                               search_mode="grid", n_candidates=200, metric="f1")

    with torch.no_grad():
        e_te = model.get_score_energy(torch.from_numpy(xte).to(device)).cpu().numpy()

    pred = (e_te > th).astype(int)

    summary = {
        "model": "cleanqvae", "dataset": args.dataset,
        "internal_test": {
            "threshold": th,
            "pr_auc": float(average_precision_score(yte, e_te)),
            "roc_auc": float(roc_auc_score(yte, e_te)),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "f1": float(f1_score(yte, pred, zero_division=0)),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("RESULT\n%s", json.dumps(summary, indent=2))

    # Visualizations
    cm = confusion_matrix(yte, pred)
    viz_list = VIZ_CONFIG[args.dataset]
    if "score_dist" in viz_list:
        plot_score_distribution(e_te, yte, th, summary["internal_test"], cm,
                                out / "score_distribution.png",
                                title=f"cleanqvae on {args.dataset} energy by label",
                                label_low="anomaly", label_normal="normal")
    if "pr_roc" in viz_list:
        plot_pr_roc_curve(yte, e_te, out / "pr_roc_curve.png",
                          title=f"cleanqvae on {args.dataset}",
                          label="cleanqvae")
    if "tsne" in viz_list and hasattr(model, "encoder"):
        with torch.no_grad():
            z = model.encoder(torch.from_numpy(xte).to(device)).cpu().numpy()
        rng = np.random.RandomState(42)
        n_show = min(5000, len(z))
        idx = rng.choice(len(z), n_show, replace=False)
        plot_latent_tsne(z[idx], yte[idx], out / "latent_tsne.png",
                         title=f"cleanqvae on {args.dataset} latent t-SNE (n={n_show})")
    if "timeseries" in viz_list:
        time = xte[:, 0]
        plot_timeseries_anomaly(time, e_te, yte, th, out / "timeseries_anomaly.png",
                                title=f"cleanqvae on {args.dataset} energy vs time")


if __name__ == "__main__":
    main()
