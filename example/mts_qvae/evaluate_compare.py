from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from _path_setup import setup_paths

setup_paths()

from mts_data import load_mts_pickles  # noqa: E402
from mts_tokenizer import MTSTokenizer  # noqa: E402
from models_mts import VAEBaseline  # noqa: E402


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "src" / "kaiwu").exists():
            return p
    return start.parent




def load_vae_ckpt(path: Path, device: torch.device) -> VAEBaseline:
    # Note: weights_only=False is required because the checkpoint contains
    # config dict that may have Path objects. We trust our own checkpoints.
    # For untrusted checkpoints, convert config to use only primitive types.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = VAEBaseline().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_qvae_ckpt(path: Path, device: torch.device):
    from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
    from kaiwu.torch_plugin.qvae import QVAE  # noqa: E402
    # Note: weights_only=False is required because the checkpoint contains
    # config dict that may have Path objects. We trust our own checkpoints.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config")
    if not cfg:
        raise ValueError("QVAE checkpoint missing config; re-train with provided script.")

    # rebuild modules
    from models_mts import MTSQVAEEncoder, MTSQVAEDecoderLogits  # local import

    latent_dim = int(cfg["latent_dim"])
    num_vis = int(cfg["num_vis"])
    num_hid = int(cfg["num_hid"])
    dist_beta = float(cfg["dist_beta"])

    encoder = MTSQVAEEncoder(input_dim=1540, hidden_dim=512, latent_dim=latent_dim)
    decoder = MTSQVAEDecoderLogits(latent_dim=latent_dim, hidden_dim=512, output_dim=1540)

    bm = RestrictedBoltzmannMachine(num_visible=num_vis, num_hidden=num_hid)

    # use a deterministic random sampler for evaluation
    from train_qvae import RandomIsingSampler  # local import

    sampler = RandomIsingSampler(num_solutions=128, seed=int(cfg.get("seed", 0)))

    # mean_x is stored indirectly in QVAE buffers; but constructor requires it.
    # We store a placeholder and immediately load the state dict.
    mean_x = np.ones(1540, dtype=np.float32) * 0.5

    model = QVAE(
        encoder=encoder,
        decoder=decoder,
        bm=bm,
        sampler=sampler,
        dist_beta=dist_beta,
        mean_x=mean_x,
        num_vis=num_vis,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    model.is_training = False
    return model


@torch.no_grad()
def sample_sequences_from_vae(model: VAEBaseline, tok: MTSTokenizer, n: int, device: torch.device) -> List[str]:
    z = torch.randn(n, 32, device=device)
    probs = model.decode(z).view(n, tok.max_len, tok.vocab_size).cpu().numpy()
    return [tok.decode_argmax(p) for p in probs]


@torch.no_grad()
def sample_sequences_from_qvae(model, tok: MTSTokenizer, n: int, device: torch.device) -> List[str]:
    from kaiwu.torch_plugin.qvae_dist_util import Exponential  # noqa: E402
    # Sample z (binary) from BM prior, then apply the same exponential smoothing used in MixtureGeneric.
    z_bin = model.bm.sample(model.sampler)  # (K, latent_dim)
    if z_bin.shape[0] < n:
        # repeat if sampler returns fewer solutions than requested
        reps = int(np.ceil(n / z_bin.shape[0]))
        z_bin = z_bin.repeat(reps, 1)
    z_bin = z_bin[:n]

    # exponential smoothing beta matches dist_beta
    smooth = Exponential(model.dist_beta)
    zeta0 = smooth.sample((n, z_bin.shape[1])).to(device)
    zeta = torch.where(z_bin.to(device) == 0.0, zeta0, 1.0 - zeta0)

    logits = model.decoder(zeta).view(n, tok.max_len, tok.vocab_size)
    probs = torch.sigmoid(logits).cpu().numpy()
    return [tok.decode_argmax(p) for p in probs]


def basic_metrics(seqs: List[str], train_set: set[str]) -> Dict[str, float]:
    tok = MTSTokenizer()
    valid = [s for s in seqs if tok.is_valid_generated(s)]
    unique = list(dict.fromkeys(valid))
    identical = [s for s in unique if s in train_set]
    lengths = [len(s) for s in unique] or [0]

    return {
        "n_total": float(len(seqs)),
        "n_valid": float(len(valid)),
        "valid_rate": float(len(valid) / max(1, len(seqs))),
        "n_unique_valid": float(len(unique)),
        "unique_rate_over_valid": float(len(unique) / max(1, len(valid))),
        "n_identical_to_train": float(len(identical)),
        "identical_rate_over_unique": float(len(identical) / max(1, len(unique))),
        "len_mean": float(np.mean(lengths)),
        "len_min": float(np.min(lengths)),
        "len_max": float(np.max(lengths)),
    }


def write_fasta(seqs: List[str], path: Path, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for i, s in enumerate(seqs):
        lines.append(f">{prefix}_{i}")
        lines.append(s)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aa_freq(seqs: List[str]) -> Dict[str, float]:
    tok = MTSTokenizer()
    # amino acids only (exclude padding '0' and cleavage '$')
    aas = [c for c in tok.alphabet if c not in ["0", "$"]]
    counts = {a: 0 for a in aas}
    total = 0
    for s in seqs:
        for c in s:
            if c in counts:
                counts[c] += 1
                total += 1
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: float(v / total) for k, v in counts.items()}


def maybe_make_plots(out_dir: Path, vae_valid_unique: List[str], qvae_valid_unique: List[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped (matplotlib not available): {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    def lens(seqs: List[str]) -> List[int]:
        return [len(s) for s in seqs] or [0]

    # Length histogram
    plt.figure(figsize=(8, 4))
    plt.hist(lens(vae_valid_unique), bins=30, alpha=0.6, label="VAE", density=True)
    plt.hist(lens(qvae_valid_unique), bins=30, alpha=0.6, label="QVAE", density=True)
    plt.xlabel("Sequence length")
    plt.ylabel("Density")
    plt.title("Length distribution (valid & unique)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "length_hist.png", dpi=160)
    plt.close()

    # Amino acid composition bar plot
    vae_f = aa_freq(vae_valid_unique)
    qvae_f = aa_freq(qvae_valid_unique)
    aas = list(vae_f.keys())
    x = np.arange(len(aas))
    width = 0.45
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, [vae_f[a] for a in aas], width=width, label="VAE")
    plt.bar(x + width / 2, [qvae_f[a] for a in aas], width=width, label="QVAE")
    plt.xticks(x, aas)
    plt.ylabel("Frequency")
    plt.title("Amino-acid composition (valid & unique)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "aa_composition.png", dpi=160)
    plt.close()


def maybe_plot_external_category_distribution(out_dir: Path, ext_name: str, vae_summary: dict, qvae_summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    vae_frac = (vae_summary or {}).get("category_fractions") or {}
    qvae_frac = (qvae_summary or {}).get("category_fractions") or {}

    cats = sorted(set(vae_frac.keys()) | set(qvae_frac.keys()))
    if not cats:
        return

    x = np.arange(len(cats))
    width = 0.45
    plt.figure(figsize=(max(10, len(cats) * 0.6), 4))
    plt.bar(x - width / 2, [vae_frac.get(c, 0.0) for c in cats], width=width, label="VAE")
    plt.bar(x + width / 2, [qvae_frac.get(c, 0.0) for c in cats], width=width, label="QVAE")
    plt.xticks(x, cats, rotation=45, ha="right")
    plt.ylabel("Fraction")
    plt.title(f"{ext_name}: predicted category fractions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{ext_name.lower()}_category_fractions.png", dpi=160)
    plt.close()


def _substitute_tokens(tokens: List[str], mapping: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        for k, v in mapping.items():
            t = t.replace("{" + k + "}", v)
        out.append(t)
    return out


def run_external_tool(name: str, cmd_tokens: List[str], mapping: Dict[str, str], cwd: Path) -> int:
    if not cmd_tokens:
        return 0
    exe = cmd_tokens[0]
    if shutil.which(exe) is None and not Path(exe).exists():
        raise FileNotFoundError(f"{name} executable not found: {exe}")

    cmd = _substitute_tokens(cmd_tokens, mapping)
    print("\n[external] " + name)
    print("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.stdout:
        print(p.stdout)
    if p.stderr:
        print(p.stderr, file=sys.stderr)
    return int(p.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline VAE vs QVAE on basic MTS generation metrics")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vae-ckpt", type=Path, required=True)
    parser.add_argument("--qvae-ckpt", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("example/mts_qvae/outputs/compare_metrics.json"))
    parser.add_argument("--save-fasta", action="store_true", help="Write sampled sequences to FASTA files.")
    parser.add_argument("--plots", action="store_true", help="Write extra plots to outputs/figs.")

    # Optional: external tool integration.
    # We DO NOT provide defaults (to avoid inventing CLI). You must pass explicit command tokens.
    # Supported placeholders inside tokens: {in_fasta}, {out_dir}, {out_path}
    parser.add_argument(
        "--targetp-cmd",
        nargs="+",
        default=None,
        help="TargetP command tokens (no defaults). Example: conda run -n targetp2 targetp ...",
    )
    parser.add_argument(
        "--deeploc-cmd",
        nargs="+",
        default=None,
        help="DeepLoc2 command tokens (no defaults).",
    )

    parser.add_argument(
        "--targetp-out",
        type=Path,
        default=None,
        help="Path to TargetP output file (if the command writes a file you want parsed).",
    )
    parser.add_argument(
        "--deeploc-out",
        type=Path,
        default=None,
        help="Path to DeepLoc output file (if the command writes a file you want parsed).",
    )
    parser.add_argument(
        "--external-schema",
        type=Path,
        default=None,
        help=(
            "JSON schema describing output columns for parsing (see external_schema.template.json). "
            "If omitted, the parser will try a best-effort inference for table outputs."
        ),
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    for name in ["data_root", "vae_ckpt", "qvae_ckpt", "out"]:
        p = getattr(args, name)
        if isinstance(p, Path) and (not p.is_absolute()):
            setattr(args, name, (repo_root / p).resolve())

    for name in ["targetp_out", "deeploc_out", "external_schema"]:
        p = getattr(args, name)
        if isinstance(p, Path) and (not p.is_absolute()):
            setattr(args, name, (repo_root / p).resolve())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = MTSTokenizer()

    train, _ = load_mts_pickles(args.data_root)
    train_set = set(train.sequences)

    vae = load_vae_ckpt(args.vae_ckpt, device)
    qvae = load_qvae_ckpt(args.qvae_ckpt, device)

    vae_seqs = sample_sequences_from_vae(vae, tok, args.n_samples, device)
    qvae_seqs = sample_sequences_from_qvae(qvae, tok, args.n_samples, device)

    # Build valid+unique lists for plots and exports
    def valid_unique(seqs: List[str]) -> List[str]:
        valid = [s for s in seqs if tok.is_valid_generated(s)]
        return list(dict.fromkeys(valid))

    vae_vu = valid_unique(vae_seqs)
    qvae_vu = valid_unique(qvae_seqs)

    metrics = {
        "vae": basic_metrics(vae_seqs, train_set=train_set),
        "qvae": basic_metrics(qvae_seqs, train_set=train_set),
        "extra": {
            "vae_aa_freq": aa_freq(vae_vu),
            "qvae_aa_freq": aa_freq(qvae_vu),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # Optional exports
    out_dir = args.out.parent
    if args.save_fasta:
        write_fasta(vae_vu, out_dir / "vae_samples.fasta", prefix="vae")
        write_fasta(qvae_vu, out_dir / "qvae_samples.fasta", prefix="qvae")
        print(f"Wrote: {out_dir / 'vae_samples.fasta'}")
        print(f"Wrote: {out_dir / 'qvae_samples.fasta'}")

    if args.plots:
        maybe_make_plots(out_dir / "figs", vae_vu, qvae_vu)
        print(f"Wrote plots under: {out_dir / 'figs'}")

    # Optional external tool execution (raw run; parsing is deliberately not assumed).
    if args.targetp_cmd or args.deeploc_cmd:
        if not args.save_fasta:
            # External tools generally require FASTA input.
            write_fasta(vae_vu, out_dir / "vae_samples.fasta", prefix="vae")
            write_fasta(qvae_vu, out_dir / "qvae_samples.fasta", prefix="qvae")

        mapping_vae = {
            "in_fasta": str((out_dir / "vae_samples.fasta").resolve()),
            "out_dir": str(out_dir.resolve()),
            "out_path": str((out_dir / "vae_external.out").resolve()),
        }
        mapping_qvae = {
            "in_fasta": str((out_dir / "qvae_samples.fasta").resolve()),
            "out_dir": str(out_dir.resolve()),
            "out_path": str((out_dir / "qvae_external.out").resolve()),
        }

        if args.targetp_cmd:
            run_external_tool("TargetP", args.targetp_cmd, mapping_vae, cwd=out_dir)
            run_external_tool("TargetP", args.targetp_cmd, mapping_qvae, cwd=out_dir)
        if args.deeploc_cmd:
            run_external_tool("DeepLoc", args.deeploc_cmd, mapping_vae, cwd=out_dir)
            run_external_tool("DeepLoc", args.deeploc_cmd, mapping_qvae, cwd=out_dir)

    # Optional: parse external outputs (if user provided them or if the external tools wrote them).
    # NOTE: We do not assume a specific DTU format; schema can be provided via --external-schema.
    if args.targetp_out or args.deeploc_out:
        from external_eval import load_output, load_schema, infer_schema_from_table, summarize_table

        schema = None
        if args.external_schema and args.external_schema.exists():
            schema = load_schema(args.external_schema)

        external_results = {}
        for ext_name, out_path in [("TargetP", args.targetp_out), ("DeepLoc", args.deeploc_out)]:
            if out_path is None:
                continue
            if not out_path.exists():
                print(f"External output missing (skip parse): {out_path}")
                continue

            obj = load_output(out_path)
            if hasattr(obj, "columns"):
                df = obj
                use_schema = schema or infer_schema_from_table(df)
                external_results[ext_name.lower()] = summarize_table(df, use_schema)
            else:
                # JSON output: store raw keys only (no assumptions)
                external_results[ext_name.lower()] = {
                    "format": "json",
                    "top_level_type": type(obj).__name__,
                }

        if external_results:
            # merge into metrics and rewrite
            metrics["external"] = external_results
            args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Updated with external summaries: {args.out}")

            # Plotting for external tools requires per-model outputs (vae vs qvae).
            # Once you provide separate output paths, we can add paired plots.

    return 0


if __name__ == "__main__":
    sys.exit(main())
