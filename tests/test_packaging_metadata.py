"""Guards for packaging metadata and module importability.

``requires-python`` in pyproject.toml decides who pip will offer the
package to; the pinned dependencies and the PEP 604 annotations in the
source decide who can actually run it. The tests here keep the three in
step:

- the declared floor must not sit below the floor required by any pinned
  dependency's own metadata, otherwise installs resolve dependencies the
  package cannot use;
- the declared floor must not sit below the documented and CI baseline
  (Python 3.10);
- every module under ``kaiwu.torch_plugin`` must import cleanly, so an
  import-time regression in any module (including rarely imported ones)
  fails the suite instead of only failing in user code;
- the PEP 604 annotations in ``gbrbm`` must stay lazy (string form),
  matching the other modules that use them.
"""

import importlib
import importlib.metadata
import os
import re
import sys
import unittest

try:
    import tomllib
except ImportError:  # Python 3.10: tomli ships with pytest
    import tomli as tomllib

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

import kaiwu

# Extend namespace package __path__ so that local src is preferred
kaiwu.__path__ = list(kaiwu.__path__) + [os.path.join(src_root, "kaiwu")]

for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")


def _declared_python_floor():
    """Read the requires-python floor from pyproject.toml.

    Returns:
        tuple[int, int]: The declared minimum (major, minor) version.
    """
    pyproject = os.path.join(src_root, "..", "pyproject.toml")
    with open(os.path.abspath(pyproject), "rb") as handle:
        data = tomllib.load(handle)
    spec = data["project"]["requires-python"]
    match = re.match(r">=\s*(\d+)\.(\d+)", spec)
    assert match, f"unsupported requires-python spec: {spec!r}"
    return (int(match.group(1)), int(match.group(2)))


def _dependency_floor(distribution):
    """Read the requires-python floor of an installed distribution.

    Args:
        distribution (str): Installed distribution name, e.g. ``"numpy"``.

    Returns:
        tuple[int, int] or None: Its minimum (major, minor) version, or
        ``None`` when the distribution declares no floor.
    """
    spec = importlib.metadata.metadata(distribution).get("Requires-Python")
    if not spec:
        return None
    match = re.match(r">=\s*(\d+)\.(\d+)", spec)
    return (int(match.group(1)), int(match.group(2))) if match else None


class TestPythonFloor(unittest.TestCase):
    """requires-python must match what the code and pins can run on."""

    def test_floor_matches_pinned_dependencies(self):
        """The declared floor must not undercut any pinned dependency."""
        floor = _declared_python_floor()
        for distribution in ("numpy", "torch", "kaiwu"):
            dep_floor = _dependency_floor(distribution)
            if dep_floor is None:
                continue
            self.assertGreaterEqual(
                floor,
                dep_floor,
                msg=(
                    f"requires-python {floor} is below {distribution}'s own "
                    f"floor {dep_floor}; installs on older interpreters "
                    "cannot resolve the pinned dependencies"
                ),
            )

    def test_floor_matches_documented_baseline(self):
        """README, installation guide, and CI all standardize on 3.10."""
        self.assertGreaterEqual(
            _declared_python_floor(),
            (3, 10),
            msg="requires-python sits below the documented 3.10 baseline",
        )


class TestModuleImportability(unittest.TestCase):
    """Every module under kaiwu.torch_plugin must import cleanly."""

    def test_all_torch_plugin_modules_importable(self):
        """Import every .py file in the package, including subpackages."""
        package_root = os.path.join(src_root, "kaiwu", "torch_plugin")
        module_names = []
        for dirpath, _dirnames, filenames in os.walk(package_root):
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                relative = os.path.relpath(os.path.join(dirpath, filename), src_root)
                parts = os.path.splitext(relative)[0].replace(os.sep, ".").split(".")
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                module_names.append(".".join(parts))
        self.assertGreaterEqual(len(module_names), 10)
        for name in sorted(module_names):
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_gbrbm_annotations_are_lazy(self):
        """PEP 604 annotations in gbrbm must stay in string form.

        ``gbrbm`` annotates parameters with ``torch.Tensor | None``. Without
        ``from __future__ import annotations`` those expressions evaluate at
        def time and require PEP 604 runtime support (Python 3.10+), which
        made the module unimportable on older interpreters even when it was
        loaded directly from source. The other PEP 604 users in the package
        (qdiffusion, maifs) already carry the future import.
        """
        module = importlib.import_module("kaiwu.torch_plugin.gbrbm")
        annotations = (
            module.GaussianBernoulliRestrictedBoltzmannMachine.gibbs_sample.__annotations__
        )
        self.assertTrue(annotations)
        for value in annotations.values():
            self.assertIsInstance(
                value,
                str,
                msg="gbrbm annotations must stay lazy "
                "(keep 'from __future__ import annotations')",
            )


if __name__ == "__main__":
    unittest.main()
