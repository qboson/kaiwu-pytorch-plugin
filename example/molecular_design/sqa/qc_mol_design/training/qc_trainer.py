"""
QC-assisted CRBM training via Simulated Annealing.

From the paper (Methods – QC-assisted training):
  "We exploit quantum annealers to model p_θ(y, h | f) and draw samples from
   them to train the constructed energy-based model."
  "For each update step of the model parameters, several samples are drawn
   from p(y,h|f) by performing multiple annealing runs."

Available reproduction proxy used here:
  D-Wave quantum annealing  →  D-Wave's Neal simulated annealing

This backend is explicitly a classical proxy and must not be reported as the
paper's D-Wave Advantage experiment (20 microseconds per annealing run).

Training algorithm:
  For each batch (f, y_pos):
    1. Positive phase: h̄_pos = P(h | y_pos, f)   [mean-field, exact]
    2. Negative phase via SA:
       a. Build per-sample QUBO Q from CRBM parameters and each f_i
       b. Sample (y_neg, h_neg) ← SA(Q)    [joint sampling of visible + hidden]
    3. Compute CD-like updates using SA samples as the negative phase:
       ΔW  = (y_pos ⊗ h̄_pos − y_neg ⊗ h_neg) / B

Per-sample mode (default, paper-faithful):
  For each sample in the batch (up to max_per_sample), build a separate QUBO
  from that sample's fingerprint. This matches the paper's approach.

Batch-mean mode (faster, legacy fallback):
  Build one QUBO using the batch-mean fingerprint. Less accurate but faster.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from qc_mol_design.config import CFG
from qc_mol_design.models.crbm import CRBM
from qc_mol_design.models.graph_conv import NeuralFingerprint
from qc_mol_design.models.predictors import MultiPropertyPredictor
from qc_mol_design.training.cd_trainer import (
    _reconstruction_free_energy_difference,
    _train_property_predictor,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SA-based negative phase sampler
# ─────────────────────────────────────────────────────────────────────────────

def _solve_qubo_neal(Q_np: np.ndarray, n_reads: int, n_sweeps: int):
    """Solve QUBO using D-Wave Neal, return list of sample dicts."""
    import neal
    n = Q_np.shape[0]
    Q_dict = {
        (i, j): float(Q_np[i, j])
        for i in range(n)
        for j in range(i, n)
        if Q_np[i, j] != 0.0
    }
    sampler  = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q_dict, num_reads=n_reads, num_sweeps=n_sweeps)
    return list(response.samples())


def _sample_neg_phase_sa(
    crbm:    CRBM,
    f_batch: torch.Tensor,
    n_reads: int = 5,
    n_sweeps: int = 500,
    per_sample: bool = True,
    max_per_sample: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw SA samples and retain each sample's conditioning index."""
    if not per_sample:
        raise ValueError("batch-mean QC sampling is not valid in strict mode")

    batch_size = f_batch.shape[0]
    n_vis = CFG.n_vis
    n_hid = CFG.n_hidden
    n_tot = n_vis + n_hid
    n_solve = batch_size if max_per_sample <= 0 else min(batch_size, max_per_sample)

    y_neg_list: List[np.ndarray] = []
    h_neg_list: List[np.ndarray] = []
    condition_indices: List[int] = []

    for condition_idx in range(n_solve):
        f_single = f_batch[condition_idx]
        Q_tensor = crbm.build_neg_phase_qubo(
            f_single,
            onehot_penalty=getattr(CFG, "qc_onehot_penalty", 0.0),
        )
        samples = _solve_qubo_neal(Q_tensor.numpy(), n_reads, n_sweeps)
        for sample in samples[:n_reads]:
            x = np.array(
                [sample.get(i, 0) for i in range(n_tot)], dtype=np.float32
            )
            # Equations (6)-(10) require expectations of the actual annealer
            # samples.  Projecting or reconstructing a returned state changes
            # the sampled p_theta(y,h|f) statistics.
            y_neg_list.append(x[:n_vis])
            h_neg_list.append(x[n_vis:])
            condition_indices.append(condition_idx)

    if not y_neg_list:
        raise RuntimeError("QC negative-phase sampler returned no samples")

    return (
        torch.FloatTensor(np.stack(y_neg_list)),
        torch.FloatTensor(np.stack(h_neg_list)),
        torch.LongTensor(condition_indices),
    )


def _qc_step(
    crbm:     CRBM,
    y0:       torch.Tensor,   # (B, n_vis)
    f:        torch.Tensor,   # (B, fp_dim)
    lr:       float,
    clip:     float = 1.0,
    n_reads:  int   = 5,
    n_sweeps: int   = 500,
    per_sample:     bool  = True,
    max_per_sample: int   = 10,
) -> float:
    """
    One QC-assisted in-place update step.
    Returns mean free energy (for monitoring).
    """
    # Negative samples carry the index of the fingerprint that generated
    # them. Positive statistics are repeated for that same condition/read.
    with torch.no_grad():
        y_neg, h_neg, condition_idx = _sample_neg_phase_sa(
            crbm, f.detach().cpu(), n_reads, n_sweeps,
            per_sample=per_sample, max_per_sample=max_per_sample,
        )
        condition_idx = condition_idx.to(f.device)
        y_neg = y_neg.to(f.device)
        h_neg = h_neg.to(f.device)
        y_pos = y0.index_select(0, condition_idx)
        f_used = f.index_select(0, condition_idx)
        ph0 = crbm.p_h_given_yf(y_pos, f_used)

        fe_difference = (
            crbm.free_energy(y_pos, f_used)
            - crbm.free_energy(y_neg, f_used)
        ).mean().item()

    batch_effective = float(y_neg.shape[0])
    with torch.no_grad():
        dW = (
            y_pos.t() @ ph0 / batch_effective
            - y_neg.t() @ h_neg / batch_effective
        )
        db = (y_pos - y_neg).mean(0)
        dc = (ph0 - h_neg).mean(0)
        dW1 = f_used.t() @ (y_pos - y_neg) / batch_effective
        dW2 = f_used.t() @ (ph0 - h_neg) / batch_effective

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


# ─────────────────────────────────────────────────────────────────────────────
# Full QC trainer
# ─────────────────────────────────────────────────────────────────────────────

class QCTrainer:
    """
    Trains the CRBM using an SA proxy for the negative phase, then trains the
    separate multi-property predictor on the frozen latent representation.
    """

    def __init__(
        self,
        gcn:  NeuralFingerprint,
        crbm: CRBM,
        pred: MultiPropertyPredictor,
    ) -> None:
        self.gcn  = gcn.to(CFG.device).eval()
        self.crbm = crbm.to(CFG.device)
        self.pred = pred.to(CFG.device)
        self.pred_optim = torch.optim.Adam(
            pred.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        self.history: List[Dict] = []
        self.predictor_history: List[Dict] = []

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        tag: str = "QC",
        epoch_callback: Optional[Callable[[int, List[Dict]], None]] = None,
    ) -> List[Dict]:
        """
        Train with QC-assisted SA and validation-based early stopping.

        Paper (p.12): "The training for all models is terminated when the
        validation metrics fail to improve over consecutive steps."
        """
        print(f"\n[{tag}] Pre-computing fingerprints ...")
        train_batches = []
        self.gcn.eval()
        with torch.no_grad():
            for X, adj, deg, y_vec, props in train_loader:
                X, adj, deg = (t.to(CFG.device) for t in (X, adj, deg))
                train_batches.append((self.gcn(X, adj, deg).cpu(), y_vec, props))

        # Pre-compute validation fingerprints
        valid_batches = []
        with torch.no_grad():
            for X, adj, deg, y_vec, props in valid_loader:
                X, adj, deg = (t.to(CFG.device) for t in (X, adj, deg))
                valid_batches.append((self.gcn(X, adj, deg).cpu(), y_vec, props))

        print(f"[{tag}] Training CRBM (QC-assisted SA) for {CFG.epochs} epochs ...")

        best_val_difference = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, CFG.epochs + 1):
            self.crbm.train()
            fe_differences = []

            for fp_batch, y_batch, _prop_batch in train_batches:
                f = fp_batch.to(CFG.device)
                y = y_batch.to(CFG.device)

                difference = _qc_step(
                    self.crbm, y, f, CFG.lr, CFG.grad_clip,
                    n_reads=CFG.qc_sa_reads, n_sweeps=CFG.qc_sa_sweeps,
                    per_sample=getattr(CFG, 'qc_per_sample', True),
                    max_per_sample=getattr(CFG, 'qc_max_per_sample', 0),
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
