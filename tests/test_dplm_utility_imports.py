"""Pure DPLM file and quality helpers must not require model dependencies."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_pure_helpers_without_site_packages(tmp_path):
    """Read/write FASTA and score sequences in a standard-library-only process."""
    script = """
import sys
from pathlib import Path
from example.qdiffusion.dplm.utils.io import read_fasta_records, save_json
from example.qdiffusion.dplm.utils.metrics import evaluate_generation_quality

path = Path(sys.argv[1])
path.write_text('>reference\\nACDE\\n', encoding='utf-8')
records = read_fasta_records(path)
summary = evaluate_generation_quality(records, records, label='self')
assert summary.identity_to_reference_mean == 1.0
assert summary.amino_acid_jsd == 0.0
save_json(path.with_suffix('.json'), {'identity': summary.identity_to_reference_mean})
assert not any(name in sys.modules for name in ('torch', 'transformers', 'omegaconf'))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(tmp_path / "input.fasta")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "input.json").read_text(encoding="utf-8").strip().endswith("}")


def test_unknown_utility_attribute_is_attribute_error():
    """Introspection does not accidentally initialize a model backend."""
    script = """
from example.qdiffusion.dplm import utils
assert utils.__all__ == ['build_dplm_qdiffusion', 'build_qdiffusion']
assert not hasattr(utils, 'not_a_builder')
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_factory_exports_keep_original_objects():
    """Installed model dependencies still expose the original factory functions."""
    pytest.importorskip("transformers")
    pytest.importorskip("omegaconf")
    utils = importlib.import_module("example.qdiffusion.dplm.utils")
    dplm_builder = importlib.import_module("example.qdiffusion.dplm.utils.dplm_builder")

    assert utils.build_qdiffusion is dplm_builder.build_qdiffusion
    assert utils.build_dplm_qdiffusion is dplm_builder.build_dplm_qdiffusion
