# -*- coding: utf-8 -*-
"""trainer: N-epoch loop with tqdm and structured logging."""
from __future__ import annotations
import numpy as np
import torch
from tqdm import tqdm

from .tuner import AnomalyTuner
from model import Config
from utils.logging import get_logger
from utils.exception import ValueError, SizeError


def set_seed(seed):
    """Seed numpy, torch CPU, and CUDA RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, cfg: Config, Xtr, ytr, Xval, yval, tag="model", log_file=None):
    """Train ``model`` for ``cfg.epochs`` and return the best-epoch state_dict.

    Each epoch runs one forward/backward pass via :class:`AnomalyTuner`,
    tracks validation PR-AUC (energy score), and checkpoints the best epoch.
    Training loss and val metrics are logged via ``utils.logging.get_logger``.

    Args:
        model: CleanEnergyQVAE instance.
        cfg:   Config with epochs / batch_size / lr / kl_beta.
        Xtr, ytr: training features and labels (numpy arrays).
        Xval, yval: validation features and labels.
        tag:   name prefix in log lines.
        log_file: optional path to also append logs to a file.

    Returns:
        state_dict of the best-epoch model (already on CPU).
    """
    logger = get_logger("trainer", log_file=log_file)

    # Validate inputs
    if cfg.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {cfg.epochs}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {cfg.batch_size}")
    if len(Xtr) != len(ytr):
        raise SizeError(f"Xtr and ytr length mismatch: {len(Xtr)} vs {len(ytr)}")
    if len(Xval) != len(yval):
        raise SizeError(f"Xval and yval length mismatch: {len(Xval)} vs {len(yval)}")

    set_seed(cfg.seed)
    tuner = AnomalyTuner(device=cfg.device)
    tuner.register_model(model)
    tuner.register_optimizer(torch.optim.Adam(model.parameters(), lr=cfg.lr))

    best = {"epoch": -1, "val_pr": -1.0, "state": None}
    pbar = tqdm(range(1, cfg.epochs + 1), desc=tag, ncols=100)
    for epoch in pbar:
        loss = tuner.train_epoch(Xtr, ytr, cfg.batch_size, kl_beta=cfg.kl_beta)
        pr_e = tuner.eval_pr(Xval, yval, score_type="energy")
        pr_r = tuner.eval_pr(Xval, yval, score_type="recon")
        pr_c = tuner.eval_pr(Xval, yval, score_type="combined")
        if pr_e > best["val_pr"]:
            best.update(epoch=epoch, val_pr=pr_e,
                        state={k: v.clone() for k, v in model.state_dict().items()})
        pbar.set_postfix(loss=f"{loss:.3f}", valPR=f"{pr_e:.4f}")
        logger.info(
            "[%s] epoch %d/%d loss=%.4f valPR(energy)=%.4f recon=%.4f comb=%.4f",
            tag, epoch, cfg.epochs, loss, pr_e, pr_r, pr_c
        )

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    logger.info("[%s] best epoch=%d valPR=%.4f", tag, best["epoch"], best["val_pr"])
    return model, best
