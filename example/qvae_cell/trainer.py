import os
import random

import kaiwu as kw
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.cim import CIMOptimizer, PrecisionReducer
from kaiwu.torch_plugin import BoltzmannMachine

from models import CellQVAE, QVAEDecoder, QVAEEncoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    metric_keys = ("loss", "neg_elbo", "kl", "recon_loss", "bm_loss")

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.n_batches = 0

    def adata_to_array(self, adata):
        x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        x = np.asarray(x, dtype=np.float32)
        if self.args.loss_type == "mse":
            return x

        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        if x_min < 0.0 or x_max > 1.0:
            x = (x - x_min) / (x_max - x_min + 1e-8)
        return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)

    def batch_indices(self, adata, batch_key):
        batch_categories = adata.obs[batch_key].astype("category")
        self.n_batches = len(batch_categories.cat.categories)
        return batch_categories.cat.codes.to_numpy()

    def make_loaders(self, x, batch_indices):
        train_idx, val_idx = train_test_split(
            np.arange(x.shape[0]),
            test_size=self.args.val_percentage,
            random_state=self.args.seed,
        )
        x_tensor = torch.tensor(x, dtype=torch.float32)
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long)
        return (
            DataLoader(
                TensorDataset(x_tensor[train_idx], batch_tensor[train_idx]),
                batch_size=self.args.batch_size,
                shuffle=True,
            ),
            DataLoader(
                TensorDataset(x_tensor[val_idx], batch_tensor[val_idx]),
                batch_size=self.args.batch_size,
                shuffle=False,
            ),
            DataLoader(
                TensorDataset(x_tensor, batch_tensor),
                batch_size=self.args.batch_size,
                shuffle=False,
            ),
        )

    def compute_mean_x(self, train_loader):
        total = None
        count = 0
        for x, _ in train_loader:
            batch_sum = x.sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            count += x.shape[0]
        return (total / count).cpu().numpy()

    def create_sampler(self):
        if self.args.sampler_type == "sa":
            return SimulatedAnnealingOptimizer(
                initial_temperature=self.args.sa_initial_temperature,
                alpha=self.args.sa_alpha,
                cutoff_temperature=self.args.sa_cutoff_temperature,
                iterations_per_t=self.args.sa_iterations_per_t,
                size_limit=self.args.sa_size_limit,
                rand_seed=self.args.sa_rand_seed,
            )
        if self.args.sampler_type != "cim":
            raise ValueError(f"Unsupported sampler_type: {self.args.sampler_type}")

        kw.common.CheckpointManager.save_dir = self.args.tmp_dir
        sampler = CIMOptimizer(
            task_name=self.args.task_name,
            project_no=self.args.project_no,
            wait=True,
            task_mode=self.args.task_mode,
            sample_number=self.args.sample_number,
        )
        return PrecisionReducer(
            sampler,
            precision=self.args.precision,
            truncated_precision=self.args.truncated_precision,
            target_bits=self.args.target_bits,
            only_feasible_solution=False,
        )

    def create_model(self, input_dim, train_loader):
        if self.args.latent_dim != self.args.num_visible + self.args.num_hidden:
            raise ValueError("latent_dim must equal num_visible + num_hidden")

        encoder = QVAEEncoder(
            input_dim,
            self.args.hidden_dim,
            self.args.latent_dim,
            self.args.normalization_method,
        )
        decoder = QVAEDecoder(
            self.args.latent_dim + self.n_batches,
            self.args.hidden_dim,
            input_dim,
            self.args.normalization_method,
        )
        model = CellQVAE(
            encoder=encoder,
            decoder=decoder,
            bm=BoltzmannMachine(self.args.latent_dim, device=self.device),
            sampler=self.create_sampler(),
            dist_beta=self.args.dist_beta,
            mean_x=self.compute_mean_x(train_loader),
            num_vis=self.args.num_visible,
            n_batches=self.n_batches,
        ).to(self.device)
        vae_optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=self.args.lr,
        )
        bm_optimizer = torch.optim.Adam(model.bm.parameters(), lr=self.args.rbm_lr)
        return model, vae_optimizer, bm_optimizer

    def run_epoch(self, model, loader, vae_optimizer, bm_optimizer, train=True):
        model.train(train)
        totals = {key: 0.0 for key in self.metric_keys}

        for x, batch_idx in loader:
            x = x.to(self.device)
            batch_idx = batch_idx.to(self.device)
            if train:
                vae_optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                loss, kl, recon_loss, q = model.cell_loss(
                    x, batch_idx, self.args.loss_type, self.args.kl_beta
                )
                if train:
                    loss.backward()
                    vae_optimizer.step()

            bm_loss_value = 0.0
            if train:
                bm_optimizer.zero_grad()
                bm_loss = model.bm_phase_loss(q, self.args.bm_weight_decay)
                bm_loss.backward()
                bm_optimizer.step()
                bm_loss_value = bm_loss.item()

            totals["loss"] += loss.item()
            totals["neg_elbo"] += loss.item()
            totals["kl"] += kl.item()
            totals["recon_loss"] += recon_loss.item()
            totals["bm_loss"] += bm_loss_value

        return {key: value / len(loader) for key, value in totals.items()}

    def train(self, model, train_loader, val_loader, vae_optimizer, bm_optimizer, ckpt_path):
        history = {
            f"{split}_{key}": []
            for split in ("train", "val")
            for key in self.metric_keys
        }
        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in tqdm(range(1, self.args.epochs + 1), desc="Training KPP QVAE"):
            train_metrics = self.run_epoch(model, train_loader, vae_optimizer, bm_optimizer, True)
            val_metrics = self.run_epoch(model, val_loader, vae_optimizer, bm_optimizer, False)

            for key in self.metric_keys:
                history[f"train_{key}"].append(train_metrics[key])
                history[f"val_{key}"].append(val_metrics[key])

            print(
                f"[Epoch {epoch}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"train_recon={train_metrics['recon_loss']:.4f} "
                f"val_recon={val_metrics['recon_loss']:.4f} "
                f"train_kl={train_metrics['kl']:.4f} "
                f"val_kl={val_metrics['kl']:.4f} "
                f"bm_loss={train_metrics['bm_loss']:.4f}"
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(model.state_dict(), ckpt_path)
                patience = 0
            else:
                patience += 1

            if self.args.checkpoint_every > 0 and epoch % self.args.checkpoint_every == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(os.path.dirname(ckpt_path), f"model_epoch{epoch}.pth"),
                )

            if self.args.early_stopping_patience > 0 and patience >= self.args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return history

    def extract_representation(self, model, eval_loader):
        model.eval()
        reps = []
        with torch.no_grad():
            for x, _ in eval_loader:
                q, _, zeta = model.encode_for_loss(x.to(self.device), self.args.loss_type)
                rep = q if self.args.representation == "q" else zeta
                reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)

    def compute_energy(self, model, eval_loader):
        model.eval()
        energies = []
        with torch.no_grad():
            for x, _ in eval_loader:
                energies.extend(model.energy(x.to(self.device), self.args.loss_type).cpu().numpy().tolist())
        return energies
