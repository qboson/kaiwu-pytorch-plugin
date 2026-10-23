"""Default path contract for the DPLM example I/O helpers."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIFF = os.path.join(REPO_ROOT, "example", "qdiffusion")
sys.path.insert(0, os.path.join(QDIFF, "dplm"))

from utils.io import default_fasta_path, default_outputs_root  # noqa: E402


def test_default_fasta_points_at_the_example_proteome():
    """The default FASTA lives under ``example/qdiffusion/data``."""
    path = default_fasta_path()
    assert path.name == "UP000005640_9606.fasta"
    assert path.parent.name == "data"
    assert path.parent.parent.name == "qdiffusion"


def test_default_outputs_are_placed_beside_the_example():
    """Workflow artifacts default to ``example/qdiffusion/outputs``."""
    path = default_outputs_root()
    assert path.name == "outputs"
    assert path.parent.name == "qdiffusion"
