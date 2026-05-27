"""Development-time path bootstrap for simple qdiffusion examples."""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_repo_src_on_path() -> None:
    """Adds the repository ``src`` and sibling DPLM example paths to ``sys.path``."""
    example_root = Path(__file__).resolve().parents[1]
    repo_root = example_root.parents[1]
    src_path = repo_root / "src"
    dplm_path = example_root / "dplm"

    for path in (src_path, dplm_path):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
