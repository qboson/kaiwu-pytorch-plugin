"""Development-time path bootstrap for qdiffusion_protein examples."""

from __future__ import annotations

from pathlib import Path
import sys


# Path bootstrap helper.

def ensure_repo_src_on_path() -> None:
    """Adds the repository ``src`` directory to ``sys.path`` when needed.

    This keeps the protein-case workflow examples runnable directly from a repository
    checkout without requiring an editable install first.
    """
    example_root = Path(__file__).resolve().parents[1]
    repo_root = example_root.parents[1]
    src_path = repo_root / "src"
    src_text = str(src_path)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
