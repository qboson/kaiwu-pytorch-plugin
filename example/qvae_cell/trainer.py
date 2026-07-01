# -*- coding: utf-8 -*-
"""
Cell-specific Trainer extending the base Trainer for AnnData and two-phase training.
"""

import os
import random

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models import CellQVAE


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Cell-specific trainer that handles AnnData loading, two-phase training,
    representation extraction, and energy computation.
    """

    metric_keys = ("loss", "neg_elbo", "kl", "recon_loss", "bm_loss")

    def __init__(self, config, device):
        """
        Args:
            config (Config): Configuration object.
            device (torch.device): Device to use for training.
        """
        self.config = config
        self.device = device
        self.n_batches = 0

    def adata_to_array(self, adata):
        """Convert AnnData.X to float32 array with optional normalization."""
        x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        x = np.asarray(x, dtype=np.float32)
        if self.config.loss_type == "mse":
            return x

        # Bernoulli: clip to [0, 1]
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        if x_min < 0.0 or x_max > 1.0:
            x = (x - x_min) / (x_max - x_min + 1e-8)
        return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)

    def batch_indices(self, adata, batch_key):
        """Encode batch categories as integer indices."""
        batch_categories = adata.obs[batch_key].astype("category")
        self.n_batches = len(batch_categories.cat.categories)
        return batch_categories.cat.codes.to_numpy()

    def make_loaders(self, x, batch_indices):
        """Create train/val/eval DataLoaders."""
        train_idx, val_idx = train_test_split(
            np.arange(x.shape[0]),
            test_size=self.config.val_percentage,
            random_state=self.config.seed,
        )
        x_tensor = torch.tensor(x, dtype=torch.float32)
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long)
        return (
            DataLoader(
                TensorDataset(x_tensor[train_idx], batch_tensor[train_idx]),
                batch_size=self.config.batch_size,
                shuffle=True,
            ),
            DataLoader(
                TensorDataset(x_tensor[val_idx], batch_tensor[val_idx]),
                batch_size=self.config.batch_size,
                shuffle=False,
            ),
            DataLoader(
                TensorDataset(x_tensor, batch_tensor),
                batch_size=self.config.batch_size,
                shuffle=False,
            ),
        )

    def compute_mean_x(self, train_loader):
        """Compute gene-wise mean of training data and return as torch.Tensor."""
        total = None
        count = 0
        for x, _ in train_loader:
            batch_sum = x.sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            count += x.shape[0]
        mean = total / count
        return mean

    def create_model(self, input_dim, train_loader):
        """
        Create CellQVAE model with optimizers.

        Returns:
            (model, vae_optimizer, bm_optimizer)
        """
        if self.config.latent_dim != self.config.num_visible + self.config.num_hidden:
            raise ValueError("latent_dim must equal num_visible + num_hidden")

        model = CellQVAE(
            input_dimension=input_dim,
            activation_fct=self.config.activation_fct,
            config=self.config,
            n_batches=self.n_batches,
        ).to(self.device)

        # Set dataset mean and train bias
        mean_x = self.compute_mean_x(train_loader)
        model.set_dataset_mean(torch.tensor(mean_x, device=self.device))
        model.set_train_bias(mean_x)

        # Optimizers
        vae_optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=self.config.lr,
        )
        bm_optimizer = torch.optim.Adam(
            model.bm.parameters(),
            lr=self.config.rbm_lr,
        )
        return model, vae_optimizer, bm_optimizer

    def run_epoch(self, model, loader, vae_optimizer, bm_optimizer, train=True):
        """
        Run a single epoch (training or validation) with two-phase updates.

        Returns:
            dict of averaged metrics.
        """
        model.train(train)
        totals = {key: 0.0 for key in self.metric_keys}

        for x, batch_idx in loader:
            x = x.to(self.device)
            batch_idx = batch_idx.to(self.device)

            # Phase 1: VAE update (encoder/decoder)
            vae_optimizer.zero_grad()
            recon_x, posterior, q, _ = model(x, batch_idx)
            loss = model.loss(x, recon_x, posterior)
            loss.backward()
            vae_optimizer.step()

            # Phase 2: BM update (detach q)
            bm_optimizer.zero_grad()
            bm_loss = model.bm_loss(q.detach(), self.config.bm_weight_decay)
            bm_loss.backward()
            bm_optimizer.step()

            totals["loss"] += loss.item()
            totals["neg_elbo"] += loss.item()
            totals["kl"] += model._kl_dist_from(posterior).mean().item()
            totals["recon_loss"] += (loss.item() - self.config.kl_beta * model._kl_dist_from(posterior).mean().item())
            totals["bm_loss"] += bm_loss.item()

        return {key: value / len(loader) for key, value in totals.items()}

    def train(self, model, train_loader, val_loader, vae_optimizer, bm_optimizer, ckpt_path):
        """
        Full training loop with early stopping and checkpointing.

        Returns:
            history dict.
        """
        history = {
            f"{split}_{key}": []
            for split in ("train", "val")
            for key in self.metric_keys
        }
        best_val = float("inf")
        best_state = None
        patience = 0

        progress = tqdm(range(1, self.config.epochs + 1), desc="Training CellQVAE")

        for epoch in progress:
            train_metrics = self.run_epoch(model, train_loader, vae_optimizer, bm_optimizer, True)
            val_metrics = self.run_epoch(model, val_loader, vae_optimizer, bm_optimizer, False)

            for key in self.metric_keys:
                history[f"train_{key}"].append(train_metrics[key])
                history[f"val_{key}"].append(val_metrics[key])

            # Print metrics
            if epoch % (self.config.epochs // 10) == 0 or epoch == 1:
                print(
                    f"[Epoch {epoch}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"train_recon={train_metrics['recon_loss']:.4f} "
                    f"val_recon={val_metrics['recon_loss']:.4f} "
                    f"bm_loss={train_metrics['bm_loss']:.4f}"
                )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(model.state_dict(), ckpt_path)
                patience = 0
            else:
                patience += 1

            if self.config.checkpoint_every > 0 and epoch % self.config.checkpoint_every == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(os.path.dirname(ckpt_path), f"model_epoch{epoch}.pth"),
                )

            if self.config.early_stopping_patience > 0 and patience >= self.config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return history


    def extract_representation(self, model, eval_loader):
        """Extract q or zeta representation from data."""
        model.eval()
        reps = []
        with torch.no_grad():
            for x, batch_idx in eval_loader:
                # 调用 forward，返回 (recon_x, posterior, q, zeta)
                _, _, q, zeta = model(x.to(self.device), batch_idx.to(self.device))
                rep = q if self.config.feature_type == "q" else zeta
                reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)

    def compute_energy(self, model, eval_loader):
        """Compute BM energy for each cell."""
        model.eval()
        energies = []
        with torch.no_grad():
            for x, _ in eval_loader:
                energies.extend(model.energy(x.to(self.device), self.config.loss_type).cpu().numpy().tolist())
        return energies