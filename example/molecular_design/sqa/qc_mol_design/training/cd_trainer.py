"""
Contrastive Divergence (CD-k) training for the CRBM.

Algorithm (CD-1 as used in the paper's baseline):
  1. Data phase:
     - Given observed data (f, y_0)
     - Compute mean-field hidden activation: h̄_0 = P(h=1 | y_0, f)
     - Sample: h_0 ~ Bernoulli(h̄_0)
  2. Reconstruction phase (k Gibbs steps):
     - Sample y_k ~ P(y | h_{k-1}, f)  [categorical per property group]
     - Compute h̄_k = P(h=1 | y_k, f)
     - Sample: h_k ~ Bernoulli(h̄_k)
  3. Parameter update (positive minus negative statistics):
     ΔW  = (y_0 ⊗ h̄_0 − y_k ⊗ h̄_k) / B
     Δb  = (y_0 − y_k) / B
     Δc  = (h̄_0 − h̄_k) / B
     ΔW1 = f^T (y_0 − y_k) / B
     ΔW2 = f^T (h̄_0 − h̄_k) / B

After the energy model is trained, a separate single-layer property predictor is
trained from observed-label latent representations.  Unknown validation/test
molecules follow the paper's MAP-label-then-hidden inference path.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from qc_mol_design.config import CFG
from qc_mol_design.models.crbm import CRBM
from qc_mol_design.models.graph_conv import NeuralFingerprint
from qc_mol_design.models.predictors import MultiPropertyPredictor


# ─────────────────────────────────────────────────────────────────────────────
# CD parameter update (in-place, no autograd)
# ─────────────────────────────────────────────────────────────────────────────

def _cd_step(
    crbm: CRBM,
    y0:   torch.Tensor,   # (B, n_vis)
    f:    torch.Tensor,   # (B, fp_dim)
    lr:   float,
    k:    int = 1,
    clip: float = 1.0,
) -> float:
    """
    One CD-k in-place parameter update for the CRBM.
    Returns F(y_data, f) - F(y_reconstructed, f), the quantity plotted in
    Figure 5a of the paper.
    """
    # ── Positive phase (data statistics) ──────────────────────────────
    ph0 = crbm.p_h_given_yf(y0, f)                 # (B, n_hid)
    h0  = crbm.sample_h(ph0)

    # ── Negative phase (fantasy statistics via k Gibbs steps) ─────────
    yk  = y0
    phk = ph0
    hk  = h0
    for _ in range(k):
        logits_y = crbm.p_y_given_hf_logits(hk, f)
        yk  = crbm.sample_y_from_logits(logits_y)
        phk = crbm.p_h_given_yf(yk, f)
        hk  = crbm.sample_h(phk)

    with torch.no_grad():
        fe_difference = (
            crbm.free_energy(y0, f) - crbm.free_energy(yk, f)
        ).mean().item()

    # ── Parameter gradients ────────────────────────────────────────────
    B = float(y0.shape[0])
    with torch.no_grad():
        dy  = (y0 - yk)                             # (B, n_vis)
        dh  = (ph0 - phk)                           # (B, n_hid)

        dW  = y0.t() @ ph0 / B - yk.t() @ phk / B  # (n_vis, n_hid)
        db  = dy.mean(0)                             # (n_vis,)
        dc  = dh.mean(0)                             # (n_hid,)
        dW1 = f.t() @ dy / B                         # (fp_dim, n_vis)
        dW2 = f.t() @ dh / B                         # (fp_dim, n_hid)

        # Gradient clipping by L2 norm
        for g in (dW, db, dc, dW1, dW2):
            norm = g.norm()
            if norm > clip:
                g.mul_(clip / (norm + 1e-8))

        crbm.W .data.add_(lr * dW)
        crbm.b .data.add_(lr * db)
        crbm.c .data.add_(lr * dc)
        crbm.W1.data.add_(lr * dW1)
        crbm.W2.data.add_(lr * dW2)

    return float(fe_difference)


def _multigroup_ce(logits: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged over property groups."""
    offsets = CFG.prop_offsets
    counts  = CFG.prop_bin_counts
    losses  = []
    for start, count in zip(offsets, counts):
        lg  = logits[:, start : start + count]
        tgt = y_onehot[:, start : start + count].argmax(-1)
        losses.append(F.cross_entropy(lg, tgt))
    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Full CD trainer
# ─────────────────────────────────────────────────────────────────────────────

class CDTrainer:
    """
    Trains the CRBM with contrastive divergence, then trains the separate
    multi-property predictor on the frozen latent representation.
    """

    def __init__(
        self,
        gcn:   NeuralFingerprint,
        crbm:  CRBM,
        pred:  MultiPropertyPredictor,
    ) -> None:
        self.gcn   = gcn.to(CFG.device).eval()
        self.crbm  = crbm.to(CFG.device)
        self.pred  = pred.to(CFG.device)

        self.pred_optim = torch.optim.Adam(
            pred.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        self.history: List[Dict] = []
        self.predictor_history: List[Dict] = []

    # ── Pre-compute fingerprints ─────────────────────────────────────────
    def precompute_batches(self, loader: DataLoader) -> List[tuple]:
        """Cache aligned fingerprint, label, and property batches."""
        batches = []
        self.gcn.eval()
        with torch.no_grad():
            for X, adj, deg, y_vec, props in loader:
                X, adj, deg = (t.to(CFG.device) for t in (X, adj, deg))
                batches.append((self.gcn(X, adj, deg).cpu(), y_vec, props))
        return batches

    # ── Training loop ────────────────────────────────────────────────────
    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        tag: str = "CD",
        epoch_callback: Optional[Callable[[int, List[Dict]], None]] = None,
    ) -> List[Dict]:
        """
        Run CFG.epochs training iterations with early stopping.

        Paper (p.12): "The training for all models is terminated when the
        validation metrics fail to improve over consecutive steps."

        We monitor the original-minus-reconstructed free-energy difference on
        the validation set (lower is better), as described in Methods and
        plotted in Figure 5a.
        Returns training history (list of per-epoch metric dicts).
        """
        print(f"\n[{tag}] Pre-computing fingerprints ...")
        train_batches = self.precompute_batches(train_loader)
        valid_batches = self.precompute_batches(valid_loader)
        print(f"[{tag}] Training CRBM for {CFG.epochs} epochs ...")

        best_val_difference = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, CFG.epochs + 1):
            self.crbm.train()
            fe_differences = []

            for batch_idx, (fp_batch, y_batch, prop_batch) in enumerate(train_batches):
                f = fp_batch.to(CFG.device)
                y = y_batch.to(CFG.device)
                # ── CD update for CRBM ─────────────────────────────────
                difference = _cd_step(
                    self.crbm, y, f, CFG.lr, CFG.cd_k, CFG.grad_clip
                )
                fe_differences.append(difference)

            epoch_info = {
                "epoch":     epoch,
                "free_energy_difference": float(
                    sum(fe_differences) / max(len(fe_differences), 1)
                ),
            }

            # ── Validation: original-minus-reconstructed free energy ──
            with torch.no_grad():
                val_differences = []
                for fp_batch, y_batch, _ in valid_batches:
                    f = fp_batch.to(CFG.device)
                    y = y_batch.to(CFG.device)
                    val_differences.append(
                        _reconstruction_free_energy_difference(self.crbm, y, f)
                    )
                val_difference = float(
                    sum(val_differences) / max(len(val_differences), 1)
                )
            epoch_info["val_free_energy_difference"] = val_difference

            # Early stopping check
            if val_difference < best_val_difference:
                best_val_difference = val_difference
                best_state = {
                    k: v.clone() for k, v in self.crbm.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            self.history.append(epoch_info)
            if epoch_callback is not None:
                epoch_callback(epoch, self.history)

            if epoch % 10 == 0 or patience_counter == 0:
                print(
                    f"  [{tag}] Epoch {epoch:3d}/{CFG.epochs} | "
                    f"FE difference: {epoch_info['free_energy_difference']:.4f} | "
                    f"Val FE difference: {val_difference:.4f} "
                    f"(best={best_val_difference:.4f}, "
                    f"patience={patience_counter}/{CFG.patience})"
                )

            if patience_counter >= CFG.patience:
                print(f"  [{tag}] Early stopping at epoch {epoch} "
                      f"(no improvement for {CFG.patience} epochs)")
                # Restore best weights
                if best_state is not None:
                    self.crbm.load_state_dict(best_state)
                break

        if best_state is not None:
            self.crbm.load_state_dict(best_state)
        self.predictor_history = _train_property_predictor(
            self.crbm, self.pred, self.pred_optim,
            train_batches, valid_batches, tag,
        )
        return self.history


@torch.no_grad()
def _reconstruction_free_energy_difference(
    crbm: CRBM,
    y_data: torch.Tensor,
    f: torch.Tensor,
) -> float:
    """One-step Gibbs reconstruction metric used for validation monitoring."""
    h = crbm.sample_h(crbm.p_h_given_yf(y_data, f))
    y_reconstructed = crbm.sample_y_from_logits(
        crbm.p_y_given_hf_logits(h, f)
    )
    return float(
        (crbm.free_energy(y_data, f) - crbm.free_energy(y_reconstructed, f))
        .mean()
        .item()
    )


def _train_property_predictor(
    crbm: CRBM,
    predictor: MultiPropertyPredictor,
    optimizer: torch.optim.Optimizer,
    train_batches: List[tuple],
    valid_batches: List[tuple],
    tag: str,
) -> List[Dict]:
    """Train the paper's separate predictor after freezing the energy model."""
    crbm.eval()
    best_validation = float("inf")
    best_state = None
    patience_counter = 0
    history: List[Dict] = []
    print(f"[{tag}] Training separate latent-property predictor ...")

    for epoch in range(1, CFG.epochs + 1):
        predictor.train()
        train_losses = []
        for fp_batch, y_batch, prop_batch in train_batches:
            f = fp_batch.to(CFG.device)
            y = y_batch.to(CFG.device)
            properties = prop_batch.to(CFG.device)
            with torch.no_grad():
                # Training molecules have observed labels; Methods then uses
                # MAP-label inference for unknown validation/test molecules.
                h = crbm.encode(y, f)
            optimizer.zero_grad()
            predictions = predictor(h)
            loss = sum(
                F.mse_loss(predictions[name], properties[:, index])
                for index, name in enumerate(("qed", "logp", "sas"))
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), CFG.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        predictor.eval()
        validation_losses = []
        with torch.no_grad():
            for fp_batch, y_batch, prop_batch in valid_batches:
                f = fp_batch.to(CFG.device)
                y = y_batch.to(CFG.device)
                properties = prop_batch.to(CFG.device)
                predictions = predictor(crbm.encode_unknown(f))
                validation_losses.append(float(sum(
                    F.mse_loss(predictions[name], properties[:, index])
                    for index, name in enumerate(("qed", "logp", "sas"))
                ).item()))

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        validation_loss = float(
            sum(validation_losses) / max(len(validation_losses), 1)
        )
        history.append({
            "epoch": epoch,
            "pred_loss": train_loss,
            "val_pred_loss": validation_loss,
        })
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.clone() for key, value in predictor.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 10 == 0 or patience_counter == 0:
            print(
                f"  [{tag} predictor] Epoch {epoch:3d}/{CFG.epochs} | "
                f"MSE: {train_loss:.4f} | Val MSE: {validation_loss:.4f}"
            )
        if patience_counter >= CFG.patience:
            break

    if best_state is not None:
        predictor.load_state_dict(best_state)
    return history
