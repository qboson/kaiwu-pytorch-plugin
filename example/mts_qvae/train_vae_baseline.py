from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from _path_setup import setup_paths

setup_paths()

from mts_data import load_mts_pickles, make_loader  # noqa: E402
from models_mts import VAEBaseline  # noqa: E402


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "src" / "kaiwu").exists():
            return p
    return start.parent


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    # matches MTS-VAE: BCE reduction='sum' + KLD sum
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    total_bce = 0.0
    total_kld = 0.0

    for (x,) in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)
        recon, mu, logvar, _ = model(x)
        loss, bce, kld = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        total_bce += float(bce.item())
        total_kld += float(kld.item())

    n = len(loader.dataset)
    return total / n, total_bce / n, total_kld / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    total_bce = 0.0
    total_kld = 0.0

    for (x,) in tqdm(loader, desc="valid", leave=False):
        x = x.to(device)
        recon, mu, logvar, _ = model(x)
        loss, bce, kld = vae_loss(recon, x, mu, logvar)
        total += float(loss.item())
        total_bce += float(bce.item())
        total_kld += float(kld.item())

    n = len(loader.dataset)
    return total / n, total_bce / n, total_kld / n


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline VAE for MTS (paper-aligned).")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("example/mts_qvae/outputs/vae_baseline.pt"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    if not args.data_root.is_absolute():
        args.data_root = (repo_root / args.data_root).resolve()
    if not args.out.is_absolute():
        args.out = (repo_root / args.out).resolve()

    torch.manual_seed(args.seed)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, valid = load_mts_pickles(args.data_root)
    train_loader = make_loader(train.x.float(), batch_size=args.batch_size, shuffle=True)
    valid_loader = make_loader(valid.x.float(), batch_size=args.batch_size, shuffle=False)

    model = VAEBaseline().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    log_path = out.with_suffix(".log.txt")

    with log_path.open("w", encoding="utf-8") as f:
        for epoch in range(args.epochs):
            tr, tr_bce, tr_kld = train_epoch(model, train_loader, optim, device)
            va, va_bce, va_kld = eval_epoch(model, valid_loader, device)

            line = (
                f"epoch={epoch} train_loss={tr:.6f} train_bce={tr_bce:.6f} train_kld={tr_kld:.6f} "
                f"valid_loss={va:.6f} valid_bce={va_bce:.6f} valid_kld={va_kld:.6f}\n"
            )
            print(line.strip())
            f.write(line)
            f.flush()

            if va < best_val:
                best_val = va
                safe_cfg = {
                    k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                }
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "valid_loss": va,
                        "config": safe_cfg,
                    },
                    out,
                )

    print(f"Saved best checkpoint: {out} (best_valid={best_val:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
