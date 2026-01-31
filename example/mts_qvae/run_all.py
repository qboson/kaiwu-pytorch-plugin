from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end MTS reproduction pipeline: download -> train VAE -> train QVAE -> evaluate."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("example/mts_qvae/data/zenodo"),
        help="Where Datasets.zip is stored and extracted.",
    )
    parser.add_argument("--skip-download", action="store_true")

    parser.add_argument("--vae-epochs", type=int, default=35)
    parser.add_argument("--qvae-epochs", type=int, default=35)
    parser.add_argument("--n-samples", type=int, default=1000)

    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-qvae", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if outputs exist (overwrites checkpoints/metrics).",
    )

    args = parser.parse_args()

    here = Path(__file__).resolve()
    repo_root = here.parents[2]

    scripts_dir = repo_root / "example" / "mts_qvae"
    download_py = scripts_dir / "download_datasets.py"
    train_vae_py = scripts_dir / "train_vae_baseline.py"
    train_qvae_py = scripts_dir / "train_qvae.py"
    eval_py = scripts_dir / "evaluate_compare.py"

    outputs_dir = repo_root / "example" / "mts_qvae" / "outputs"
    vae_ckpt = outputs_dir / "vae_baseline.pt"
    qvae_ckpt = outputs_dir / "qvae.pt"
    metrics_json = outputs_dir / "compare_metrics.json"

    # 1) Download + extract
    if not args.skip_download:
        run(
            [
                sys.executable,
                str(download_py),
                "--out-dir",
                str(args.data_root),
            ],
            cwd=repo_root,
        )
    else:
        print("\n[skip] download_datasets.py")

    # 2) Train VAE baseline
    if not args.skip_vae:
        if args.force and vae_ckpt.exists():
            vae_ckpt.unlink()
        if (not vae_ckpt.exists()) or args.force:
            run(
                [
                    sys.executable,
                    str(train_vae_py),
                    "--data-root",
                    str(args.data_root),
                    "--epochs",
                    str(args.vae_epochs),
                ],
                cwd=repo_root,
            )
        else:
            print(f"\n[skip] VAE checkpoint exists: {vae_ckpt}")
    else:
        print("\n[skip] train_vae_baseline.py")

    # 3) Train QVAE
    if not args.skip_qvae:
        if args.force and qvae_ckpt.exists():
            qvae_ckpt.unlink()
        if (not qvae_ckpt.exists()) or args.force:
            run(
                [
                    sys.executable,
                    str(train_qvae_py),
                    "--data-root",
                    str(args.data_root),
                    "--epochs",
                    str(args.qvae_epochs),
                ],
                cwd=repo_root,
            )
        else:
            print(f"\n[skip] QVAE checkpoint exists: {qvae_ckpt}")
    else:
        print("\n[skip] train_qvae.py")

    # 4) Evaluate
    if not args.skip_eval:
        if args.force and metrics_json.exists():
            metrics_json.unlink()
        if (not metrics_json.exists()) or args.force:
            run(
                [
                    sys.executable,
                    str(eval_py),
                    "--data-root",
                    str(args.data_root),
                    "--vae-ckpt",
                    str(vae_ckpt),
                    "--qvae-ckpt",
                    str(qvae_ckpt),
                    "--n-samples",
                    str(args.n_samples),
                    "--out",
                    str(metrics_json),
                ],
                cwd=repo_root,
            )
        else:
            print(f"\n[skip] Metrics exist: {metrics_json}")
    else:
        print("\n[skip] evaluate_compare.py")

    print("\nDone.")
    print(f"- VAE: {vae_ckpt}")
    print(f"- QVAE: {qvae_ckpt}")
    print(f"- Metrics: {metrics_json}")

    # Optional environment hints
    targetp = shutil.which("targetp") or shutil.which("targetp2")
    deeploc = shutil.which("deeploc") or shutil.which("deeploc2")
    if not targetp:
        print("\nNote: TargetP not found on PATH (optional paper-aligned evaluation).")
    if not deeploc:
        print("Note: DeepLoc not found on PATH (optional paper-aligned evaluation).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
