from __future__ import annotations

import argparse
import fnmatch
import os
import zipfile
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "src" / "kaiwu").exists():
            return p
    return start.parent


def should_exclude(rel_posix: str) -> bool:
    # Exclude large data/training artifacts
    excluded_prefixes = [
        "example/mts_qvae/data/",
        "example/mts_qvae/outputs/",
        ".venv/",
    ]
    if any(rel_posix.startswith(p) for p in excluded_prefixes):
        return True

    # Also exclude common caches
    excluded_globs = [
        "**/__pycache__/**",
        "**/.pytest_cache/**",
        "**/*.pyc",
    ]
    return any(fnmatch.fnmatch(rel_posix, g) for g in excluded_globs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small zip bundle for Issue #26 submission (code/docs only; excludes datasets/outputs)."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("example/mts_qvae/issue_26_submission_bundle.zip"),
        help="Output zip path (relative to repo root by default).",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    out = args.out
    if not out.is_absolute():
        out = (repo_root / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    include_paths = [
        repo_root / ".gitignore",
        repo_root / "train_qvae.py",
        repo_root / "example" / "mts_qvae",
    ]

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in include_paths:
            if p.is_dir():
                for root, _, files in os.walk(p):
                    for name in files:
                        fp = Path(root) / name
                        rel = fp.relative_to(repo_root).as_posix()
                        if should_exclude(rel):
                            continue
                        zf.write(fp, arcname=rel)
            else:
                rel = p.relative_to(repo_root).as_posix()
                if not should_exclude(rel):
                    zf.write(p, arcname=rel)

    print(f"Wrote bundle: {out}")
    print("Contains code/docs only (no datasets, no outputs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
