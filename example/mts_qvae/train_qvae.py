from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from _path_setup import setup_paths

setup_paths()

from mts_data import load_mts_pickles, make_loader  # noqa: E402
from models_mts import MTSQVAEEncoder, MTSQVAEDecoderLogits  # noqa: E402

from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
from kaiwu.torch_plugin.qvae import QVAE  # noqa: E402


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "src" / "kaiwu").exists():
            return p
    return start.parent


@dataclass
class SimulatedAnnealingSampler:
    """Simulated Annealing sampler for solving Ising optimization problems.

    This sampler implements the Simulated Annealing algorithm to find
    approximate solutions to the Ising model energy minimization problem.
    Unlike RandomIsingSampler, this actually solves the optimization problem.

    The Ising Hamiltonian is: H(s) = s^T * J * s
    where s ∈ {-1, +1}^N and J is the Ising matrix.

    Args:
        num_solutions: Number of independent SA runs (returns best solutions).
        num_sweeps: Number of Monte Carlo sweeps per run.
        beta_start: Initial inverse temperature (low = high T = more exploration).
        beta_end: Final inverse temperature (high = low T = more exploitation).
        seed: Random seed for reproducibility.
    """

    num_solutions: int = 64
    num_sweeps: int = 1000
    beta_start: float = 0.1
    beta_end: float = 10.0
    seed: int = 0

    def _compute_energy(self, s: np.ndarray, J: np.ndarray) -> float:
        """Compute Ising energy: E = s^T * J * s"""
        return float(s @ J @ s)

    def _single_run(self, J: np.ndarray, rng: np.random.Generator) -> tuple:
        """Single simulated annealing run."""
        n = J.shape[0]
        # Random initial state
        s = rng.integers(0, 2, size=n) * 2 - 1  # {-1, +1}
        s = s.astype(np.float64)
        
        energy = self._compute_energy(s, J)
        best_s = s.copy()
        best_energy = energy

        # Exponential cooling schedule
        betas = np.geomspace(self.beta_start, self.beta_end, self.num_sweeps)

        for beta in betas:
            # One sweep: try flipping each spin once
            for i in rng.permutation(n):
                # Compute energy delta for flipping spin i
                # ΔE = -2 * s_i * (sum_j J_ij * s_j)
                delta_e = -2.0 * s[i] * np.dot(J[i, :], s)
                
                # Metropolis acceptance
                if delta_e < 0 or rng.random() < np.exp(-beta * delta_e):
                    s[i] = -s[i]
                    energy += delta_e
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_s = s.copy()

        return best_s, best_energy

    def solve(self, ising_matrix: np.ndarray) -> np.ndarray:
        """Solve the Ising problem using Simulated Annealing.

        Args:
            ising_matrix: (N+1, N+1) Ising matrix from BoltzmannMachine.

        Returns:
            np.ndarray: Shape (num_solutions, N+1), spins in {-1, +1}.
        """
        if isinstance(ising_matrix, torch.Tensor):
            J = ising_matrix.detach().cpu().numpy()
        else:
            J = np.asarray(ising_matrix)
        
        n = J.shape[0]
        solutions = []
        energies = []

        # Run independent SA runs with different random seeds
        for i in range(self.num_solutions):
            rng = np.random.default_rng(self.seed + i)
            s, e = self._single_run(J, rng)
            solutions.append(s)
            energies.append(e)

        # Sort by energy (best first) and return
        order = np.argsort(energies)
        result = np.array([solutions[i] for i in order], dtype=np.int8)
        return result


@dataclass
class RandomIsingSampler:
    """A minimal **placeholder** sampler (FOR TESTING ONLY).

    ⚠️ WARNING: This returns RANDOM solutions, not optimized ones.
    Use SimulatedAnnealingSampler for actual training.
    """

    num_solutions: int = 64
    seed: int = 0

    def solve(self, ising_matrix: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = ising_matrix.shape[0]  # N+1
        sol = rng.integers(0, 2, size=(self.num_solutions, n), endpoint=False)
        sol = sol * 2 - 1  # -> {-1, +1}
        return sol.astype(np.int8)


def train_epoch(model: QVAE, loader, optimizer, device, kl_beta: float):
    model.train()
    model.is_training = True

    total_loss = 0.0
    total_elbo = 0.0
    total_kl = 0.0
    total_cost = 0.0

    for (x,) in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, _, neg_elbo, wd_loss, kl, cost, _, _ = model.neg_elbo(x, kl_beta)
        loss = neg_elbo + wd_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_elbo += float(neg_elbo.item())
        total_kl += float(kl.item())
        total_cost += float(cost.item())

    n = len(loader)
    return total_loss / n, total_elbo / n, total_kl / n, total_cost / n


@torch.no_grad()
def eval_epoch(model: QVAE, loader, device, kl_beta: float):
    model.eval()
    model.is_training = False

    total_loss = 0.0
    total_elbo = 0.0
    total_kl = 0.0
    total_cost = 0.0

    for (x,) in tqdm(loader, desc="valid", leave=False):
        x = x.to(device)
        _, _, neg_elbo, wd_loss, kl, cost, _, _ = model.neg_elbo(x, kl_beta)
        loss = neg_elbo + wd_loss

        total_loss += float(loss.item())
        total_elbo += float(neg_elbo.item())
        total_kl += float(kl.item())
        total_cost += float(cost.item())

    n = len(loader)
    return total_loss / n, total_elbo / n, total_kl / n, total_cost / n


def main() -> int:
    parser = argparse.ArgumentParser(description="Train QVAE on MTS one-hot (paper-aligned input).")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("example/mts_qvae/outputs/qvae.pt"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-vis", type=int, default=16)
    parser.add_argument("--num-hid", type=int, default=16)
    parser.add_argument("--dist-beta", type=float, default=10.0)
    parser.add_argument("--kl-beta", type=float, default=1.0)

    parser.add_argument("--sampler-solutions", type=int, default=64)
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["sa", "random"],
        default="sa",
        help="Sampler type: 'sa' (SimulatedAnnealing, default) or 'random' (for fast testing only)"
    )
    parser.add_argument("--sa-sweeps", type=int, default=1000, help="Number of SA sweeps per run")
    parser.add_argument("--sa-beta-start", type=float, default=0.1, help="SA initial inverse temperature")
    parser.add_argument("--sa-beta-end", type=float, default=10.0, help="SA final inverse temperature")
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    if not args.data_root.is_absolute():
        args.data_root = (repo_root / args.data_root).resolve()
    if not args.out.is_absolute():
        args.out = (repo_root / args.out).resolve()

    if args.num_vis + args.num_hid != args.latent_dim:
        raise ValueError("Require num_vis + num_hid == latent_dim for RBM-backed QVAE")

    torch.manual_seed(args.seed)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, valid = load_mts_pickles(args.data_root)
    mean_x = train.mean_x

    train_loader = make_loader(train.x.float(), batch_size=args.batch_size, shuffle=True)
    valid_loader = make_loader(valid.x.float(), batch_size=args.batch_size, shuffle=False)

    encoder = MTSQVAEEncoder(input_dim=1540, hidden_dim=512, latent_dim=args.latent_dim)
    decoder = MTSQVAEDecoderLogits(latent_dim=args.latent_dim, hidden_dim=512, output_dim=1540)

    bm = RestrictedBoltzmannMachine(num_visible=args.num_vis, num_hidden=args.num_hid)
    
    # Select sampler: SimulatedAnnealing (real solver) or Random (placeholder)
    if args.sampler == "sa":
        print(f"Using SimulatedAnnealingSampler (sweeps={args.sa_sweeps}, beta={args.sa_beta_start}→{args.sa_beta_end})")
        sampler = SimulatedAnnealingSampler(
            num_solutions=args.sampler_solutions,
            num_sweeps=args.sa_sweeps,
            beta_start=args.sa_beta_start,
            beta_end=args.sa_beta_end,
            seed=args.seed,
        )
    else:
        print("⚠️ Using RandomIsingSampler (placeholder - NOT for scientific results!)")
        sampler = RandomIsingSampler(num_solutions=args.sampler_solutions, seed=args.seed)

    model = QVAE(
        encoder=encoder,
        decoder=decoder,
        bm=bm,
        sampler=sampler,
        dist_beta=args.dist_beta,
        mean_x=mean_x,
        num_vis=args.num_vis,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    log_path = out.with_suffix(".log.txt")

    with log_path.open("w", encoding="utf-8") as f:
        for epoch in range(args.epochs):
            tr_loss, tr_elbo, tr_kl, tr_cost = train_epoch(
                model, train_loader, optim, device, kl_beta=args.kl_beta
            )
            va_loss, va_elbo, va_kl, va_cost = eval_epoch(
                model, valid_loader, device, kl_beta=args.kl_beta
            )

            line = (
                f"epoch={epoch} train_loss={tr_loss:.6f} train_neg_elbo={tr_elbo:.6f} train_kl={tr_kl:.6f} train_cost={tr_cost:.6f} "
                f"valid_loss={va_loss:.6f} valid_neg_elbo={va_elbo:.6f} valid_kl={va_kl:.6f} valid_cost={va_cost:.6f}\n"
            )
            print(line.strip())
            f.write(line)
            f.flush()

            if va_loss < best_val:
                best_val = va_loss
                safe_cfg = {
                    k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                }
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "valid_loss": va_loss,
                        "config": safe_cfg,
                    },
                    out,
                )

    print(f"Saved best checkpoint: {out} (best_valid={best_val:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
