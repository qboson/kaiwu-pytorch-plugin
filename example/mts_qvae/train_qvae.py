from __future__ import annotations

import argparse
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
class RandomIsingSampler:
    """A minimal sampler compatible with AbstractBoltzmannMachine.sample().

    The library expects `solve(ising_matrix)` returning a numpy array of shape (K, N+1)
    with spins in {-1, +1}.

    This sampler is *not* a good sampler; it is only to make the example runnable
    without external Kaiwu solvers.
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
    raise SystemExit(main())
