"""Regression test: the ESM2 evaluator's dependency must be declared.

``example/qdiffusion/dplm/workflows/esm2_eval_helpers.py`` imports ``esm`` and
exits at import time when the package is missing, yet the example requirements
never declared a distribution providing it. The advertised evaluation stage
therefore failed on a fresh environment install instead of installing ``esm``
alongside the rest of the example dependencies.
"""

from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "example" / "qdiffusion" / "requirements.txt"
EVALUATOR = (
    REPO_ROOT
    / "example"
    / "qdiffusion"
    / "dplm"
    / "workflows"
    / "esm2_eval_helpers.py"
)


def _declared_names():
    names = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        names.add(Requirement(stripped).name.lower().replace("_", "-"))
    return names


def test_esm_dependency_is_declared_for_the_qdiffusion_example():
    """The requirements must install the ``esm`` import the evaluator needs."""
    names = _declared_names()

    assert "fair-esm" in names, (
        "example/qdiffusion/requirements.txt must declare 'fair-esm' "
        "(the PyPI distribution providing the 'esm' package), because "
        "dplm/workflows/esm2_eval_helpers.py imports it at module load time"
    )


def test_declared_requirements_stay_parseable():
    """Every non-comment line is a valid requirement specifier."""
    # The parse above already raises on malformed lines; keep an explicit
    # smoke assertion so a failure names this contract.
    assert _declared_names(), "expected at least one declared requirement"


def test_evaluator_still_imports_esm_with_a_guard():
    """The dependency name matches the import the evaluator guards."""
    source = EVALUATOR.read_text(encoding="utf-8")

    assert "import esm" in source
    assert "SystemExit" in source
