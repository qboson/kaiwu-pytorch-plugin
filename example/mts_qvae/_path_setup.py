from __future__ import annotations

import sys
from pathlib import Path


def setup_paths() -> None:
    """Make example scripts runnable from repo root.

    - Adds this folder to sys.path (so `import mts_data` works)
    - Adds repo `src/` to sys.path (so `import kaiwu.torch_plugin` works without install)
    """

    here = Path(__file__).resolve().parent

    # Find repo root by walking up until `src/kaiwu` exists.
    repo_root = None
    for p in [here] + list(here.parents):
        if (p / "src" / "kaiwu").exists():
            repo_root = p
            break
    if repo_root is None:
        # Fallback: assume standard layout example/mts_qvae under repo root
        repo_root = here.parents[1]

    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    src = repo_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
