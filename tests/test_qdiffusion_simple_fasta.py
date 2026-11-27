"""Regression tests for FASTA input handling in the simple QDiffusion examples.

Both ``simple_train_example.py`` and ``simple_generate_example.py`` described
a "bundled" FASTA file that is not part of the repository, offered no way to
point at a local file, and only failed after the expensive pretrained-model
construction. The examples must validate the FASTA path before building the
generator and accept an explicit path override.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLE_DIR = REPO_ROOT / "example" / "qdiffusion" / "simple"
SCRIPTS = ["simple_train_example.py", "simple_generate_example.py"]


def _load_simple_module(monkeypatch, filename):
    """Loads one simple example with the heavy builder replaced by a tripwire."""
    builder_calls = []

    def _tripwire_build_qdiffusion(**kwargs):
        builder_calls.append(kwargs)
        raise AssertionError("model construction must not run in these tests")

    dplm_pkg = types.ModuleType("dplm")
    dplm_pkg.__path__ = [str(SIMPLE_DIR.parent / "dplm")]
    utils_pkg = types.ModuleType("dplm.utils")
    utils_pkg.__path__ = [str(SIMPLE_DIR.parent / "dplm" / "utils")]
    builder_mod = types.ModuleType("dplm.utils.dplm_builder")
    builder_mod.build_qdiffusion = _tripwire_build_qdiffusion
    for module in (dplm_pkg, utils_pkg, builder_mod):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.syspath_prepend(str(SIMPLE_DIR))

    spec = importlib.util.spec_from_file_location(
        f"simple_example_under_test.{Path(filename).stem}", SIMPLE_DIR / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, builder_calls


def _write_fasta(path, records):
    path.write_text(
        "".join(f">{header}\n{sequence}\n" for header, sequence in records),
        encoding="utf-8",
    )


@pytest.mark.parametrize("filename", SCRIPTS)
def test_examples_validate_fasta_before_building_the_model(
    monkeypatch, filename
):
    """A missing default FASTA must fail before any pretrained model is built."""
    module, builder_calls = _load_simple_module(monkeypatch, filename)

    if module.default_fasta_path().exists():
        pytest.skip("bundled proteome data present in this checkout")

    with pytest.raises(FileNotFoundError):
        module.main([])

    assert builder_calls == []


@pytest.mark.parametrize("filename", SCRIPTS)
def test_explicit_fasta_path_is_resolved(monkeypatch, tmp_path, filename):
    module, builder_calls = _load_simple_module(monkeypatch, filename)

    fasta = tmp_path / "custom.fasta"
    _write_fasta(fasta, [("seq1", "ACDEFGHIKLMNPQRSTVWY")])

    resolved = module.resolve_fasta_path(str(fasta))

    assert resolved == fasta
    assert resolved.is_file()
    assert builder_calls == []


@pytest.mark.parametrize("filename", SCRIPTS)
def test_missing_explicit_fasta_path_names_the_file(monkeypatch, tmp_path, filename):
    module, _ = _load_simple_module(monkeypatch, filename)

    missing = tmp_path / "does_not_exist.fasta"

    with pytest.raises(FileNotFoundError, match="does_not_exist.fasta"):
        module.resolve_fasta_path(str(missing))


@pytest.mark.parametrize("filename", SCRIPTS)
def test_cli_accepts_a_fasta_path_argument(monkeypatch, tmp_path, filename):
    module, _ = _load_simple_module(monkeypatch, filename)
    fasta = tmp_path / "proteome.fasta"

    args = module.parse_args(["--fasta-path", str(fasta)])

    assert args.fasta_path == str(fasta)
    assert module.parse_args([]).fasta_path is None


def test_default_fasta_path_points_at_the_documented_proteome(monkeypatch):
    module, _ = _load_simple_module(monkeypatch, "simple_train_example.py")

    default = module.default_fasta_path()

    assert default.name == "UP000005640_9606.fasta"
    assert default.parent.name == "data"
