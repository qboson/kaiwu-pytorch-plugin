"""The README architecture diagram must list every public plugin module."""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
README_ZH = Path(__file__).resolve().parents[1] / "README_ZH.md"
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "kaiwu" / "torch_plugin"

MERMAID_RE = re.compile(r"```mermaid\n(.*?)```", re.S)
MODULE_RE = re.compile(r"([a-z][a-z0-9_]+)\.py")


def _expected_modules() -> set[str]:
    expected = set()
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        if path.stat().st_size == 0:
            continue
        expected.add(path.stem)
    return expected


def _diagram(text: str) -> str:
    match = MERMAID_RE.search(text)
    assert match is not None, "README is missing a mermaid diagram"
    return match.group(1)


def test_diagram_lists_every_public_module() -> None:
    block = _diagram(README.read_text(encoding="utf-8"))
    documented = set(MODULE_RE.findall(block))
    missing = sorted(_expected_modules() - documented)
    assert not missing, f"README diagram omits public modules: {missing}"


def test_diagram_lists_the_maifs_package() -> None:
    block = _diagram(README.read_text(encoding="utf-8"))
    assert "maifs/" in block, "README diagram omits the maifs/ package"


def test_chinese_readme_lists_the_same_modules() -> None:
    block = _diagram(README_ZH.read_text(encoding="utf-8"))
    documented = set(MODULE_RE.findall(block))
    missing = sorted(_expected_modules() - documented)
    assert not missing, f"README_ZH diagram omits public modules: {missing}"
    assert "maifs/" in block, "README_ZH diagram omits the maifs/ package"
