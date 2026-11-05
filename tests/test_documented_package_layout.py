# -*- coding: utf-8 -*-
"""The installation guide's package layout must match the shipped package.

``docs/source/getting_started/installation.md`` is the page new users read after
``pip install``; its structure tree is the map of what they actually import. Two
kinds of drift are checked here:

* every non-empty public module of ``src/kaiwu/torch_plugin`` (and the ``maifs``
  subpackage) is listed in the tree, so a new module cannot become invisible;
* every class named in the "Key Code Entities" table is a real class defined in
  the package (``GaussianBernoulliRBM`` used to be listed instead of
  ``GaussianBernoulliRestrictedBoltzmannMachine``).
"""

from __future__ import annotations

import ast
import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "source" / "getting_started" / "installation.md"
PACKAGE = REPO_ROOT / "src" / "kaiwu" / "torch_plugin"
TREE_RE = re.compile(r"## Project Structure & Package Layout\n\n```bash\n(.*?)```", re.S)
ENTITY_RE = re.compile(r"\|\s*\*\*`(\w+)`\*\*\s*\|")


def documented_tree() -> str:
    """Returns the fenced structure tree of the installation guide."""
    match = TREE_RE.search(DOC.read_text(encoding="utf-8"))
    assert match is not None, "structure tree block not found in installation.md"
    return match.group(1)


def public_modules() -> list[str]:
    """Non-empty, public, top-level modules of the package."""
    modules = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        if path.stat().st_size == 0:
            continue
        modules.append(path.name)
    return modules


def documented_classes() -> set[str]:
    """Class names written in the Key Code Entities table."""
    return set(ENTITY_RE.findall(DOC.read_text(encoding="utf-8")))


def defined_classes() -> set[str]:
    """Every class defined anywhere inside the package."""
    names: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="ignore"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                names.add(node.name)
    return names


class TestDocumentedPackageLayout(unittest.TestCase):
    """The published layout must not drift from the package."""

    def test_every_public_module_is_listed(self):
        """Each non-empty public module appears in the structure tree."""
        tree = documented_tree()
        missing = [name for name in public_modules() if name not in tree]
        self.assertEqual(missing, [], f"installation.md tree omits modules: {missing}")

    def test_maifs_subpackage_is_listed(self):
        """The MAIFS subpackage is part of the documented layout."""
        self.assertIn("maifs/", documented_tree())

    def test_documented_entities_are_real_classes(self):
        """Every class in the Key Code Entities table is defined in the package."""
        table = documented_classes()
        self.assertGreaterEqual(len(table), 4, "entity table looks empty")
        known = defined_classes()
        unknown = sorted(name for name in table if name not in known)
        self.assertEqual(unknown, [], f"entity table names classes that do not exist: {unknown}")


if __name__ == "__main__":
    unittest.main()
