"""Compatibility entrypoint.

Issue #26 discussion/README refers to running:

    python train_qvae.py --data-root example/mts_qvae/data/zenodo

This wrapper forwards to the actual implementation under example/mts_qvae.
"""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    import runpy
    import sys

    repo_root = Path(__file__).resolve().parent
    target = repo_root / "example" / "mts_qvae" / "train_qvae.py"

    # Ensure local imports like `_path_setup` resolve.
    mts_qvae_dir = target.parent
    if str(mts_qvae_dir) not in sys.path:
        sys.path.insert(0, str(mts_qvae_dir))

    # Ensure argv[0] looks like the invoked script.
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
