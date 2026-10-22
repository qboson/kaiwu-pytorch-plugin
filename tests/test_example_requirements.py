"""The example requirements must declare the modules the examples import eagerly.

``example/qvae_mnist/utils/helpers.py`` imports ``gif`` and ``imageio`` at module
level, and ``trainer/trainer.py`` imports that module, so the example dependency
list has to provide them: before this was declared, a fresh install of every
requirement file in the repository could not import the pipeline at all
(``ModuleNotFoundError: No module named 'gif'``).
"""

import ast
import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements" / "requirements_example.txt"
HELPERS = REPO_ROOT / "example" / "qvae_mnist" / "utils" / "helpers.py"


def _declared_distributions() -> set[str]:
    """Distribution names listed in the example requirements file."""
    names: set[str] = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        names.add(re.split(r"[<>=!~\[;]", line)[0].strip().lower())
    return names


def _module_level_imports(path: pathlib.Path) -> set[str]:
    """Top-level import roots of a module (lazy/guarded imports are excluded)."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class TestExampleRequirements(unittest.TestCase):
    """Eagerly imported example dependencies must be installable from the list."""

    def test_qvae_mnist_helpers_dependencies_are_declared(self):
        """gif/imageio are module-level imports of a module the training pipeline imports."""
        modules = _module_level_imports(HELPERS)
        self.assertTrue(
            {"gif", "imageio"}.issubset(modules),
            "example/qvae_mnist/utils/helpers.py no longer imports gif/imageio; "
            "update this guard together with the requirements list",
        )
        declared = _declared_distributions()
        missing = sorted(module for module in ("gif", "imageio") if module not in declared)
        self.assertEqual(
            missing, [], f"requirements_example.txt does not declare {missing}"
        )


if __name__ == "__main__":
    unittest.main()
