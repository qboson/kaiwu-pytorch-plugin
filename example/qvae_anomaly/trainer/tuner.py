# -*- coding: utf-8 -*-
"""AnomalyTuner: single-epoch train/eval."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from utils.logging import get_logger
from utils.exception import SizeError, ArgumentError

logger = get_logger("anomaly_tuner")


def make_loader(X, y, bs, shuffle):
    if bs <= 0:
        raise ArgumentError(f"batch_size must be positive, got {bs}")
    if len(X) != len(y):
        raise SizeError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=bs, shuffle=shuffle)


class AnomalyTuner:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.optimizer = None

    def register_model(self, model):
        self.model = model.to(self.device)
        logger.debug("Registered model on device=%s", self.device)

    def register_optimizer(self, optimizer):
        self.optimizer = optimizer
        logger.debug("Registered optimizer: %s", type(optimizer).__name__)

    def train_epoch(self, Xtr, ytr, batch_size, kl_beta):
        self.model.train()
        losses = []
        n_batches = 0
        for xb, yb in make_loader(Xtr, ytr, batch_size, True):
            xb = xb.to(self.device); yb = yb.to(self.device)
            terms = self.model.loss_terms(xb, kl_beta=kl_beta, labels=yb)
            self.optimizer.zero_grad()
            terms["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(terms["total_loss"].detach()))
            n_batches += 1
        avg = float(np.mean(losses))
        logger.debug("train epoch done: %d batches, avg loss=%.4f", n_batches, avg)
        return avg

    @torch.no_grad()
    def eval_pr(self, Xval, yval, score_type="energy"):
        self.model.eval()
        X = torch.from_numpy(Xval).to(self.device)
        if score_type == "energy":
            s = self.model.get_score_energy(X).cpu().numpy()
        elif score_type == "recon":
            s = -self.model.get_reconstruction_error(X).cpu().numpy()
        elif score_type == "combined":
            e = self.model.get_score_energy(X).cpu().numpy()
            r = -self.model.get_reconstruction_error(X).cpu().numpy()
            # z-score normalize and combine
            e = (e - e.mean()) / (e.std() + 1e-8)
            r = (r - r.mean()) / (r.std() + 1e-8)
            s = 0.5 * e + 0.5 * r
        else:
            raise ValueError(f"unknown score_type {score_type}")
        pr = float(average_precision_score(yval, s))
        logger.debug("eval_pr(score_type=%s) = %.4f on %d samples", score_type, pr, len(yval))
        return pr
