import os
import sys
import unittest

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

import kaiwu.torch_plugin as torch_plugin

from kaiwu.torch_plugin import (
    AbstractBoltzmannMachine,
    GaussianBernoulliRestrictedBoltzmannMachine,
)


class TestPublicApiSurface(unittest.TestCase):
    """Guard the package-root API surface documented for users."""

    def test_all_entries_resolve_to_module_attributes(self):
        """Every name in __all__ must be an attribute of the package."""
        for name in torch_plugin.__all__:
            self.assertTrue(
                hasattr(torch_plugin, name),
                f"__all__ entry {name!r} is missing from kaiwu.torch_plugin",
            )

    def test_all_entries_are_unique(self):
        """__all__ must not list the same public name twice."""
        self.assertEqual(len(torch_plugin.__all__), len(set(torch_plugin.__all__)))

    def test_all_entries_are_public_names(self):
        """Public API entries must not start with an underscore."""
        for name in torch_plugin.__all__:
            self.assertFalse(
                name.startswith("_"),
                f"__all__ entry {name!r} is not a public name",
            )

    def test_documented_key_entities_are_importable(self):
        """The Key Code Entities table in installation.md must match the API.

        docs/source/getting_started/installation.md documents
        ``AbstractBoltzmannMachine``, ``BoltzmannMachine``,
        ``RestrictedBoltzmannMachine`` and
        ``GaussianBernoulliRestrictedBoltzmannMachine`` as the key entities
        of the package, so all four must be importable from the package root.
        """
        documented_entities = (
            "AbstractBoltzmannMachine",
            "BoltzmannMachine",
            "RestrictedBoltzmannMachine",
            "GaussianBernoulliRestrictedBoltzmannMachine",
        )
        for name in documented_entities:
            self.assertIn(name, torch_plugin.__all__)
            self.assertTrue(hasattr(torch_plugin, name))

    def test_gbrbm_is_abstract_boltzmann_machine_subclass(self):
        """The exported GBRBM class must be the model class, not a helper."""
        self.assertTrue(
            issubclass(
                GaussianBernoulliRestrictedBoltzmannMachine,
                AbstractBoltzmannMachine,
            )
        )

    def test_gbrbm_deep_path_is_unchanged(self):
        """The historical deep import path must keep working."""
        from kaiwu.torch_plugin.gbrbm import (
            GaussianBernoulliRestrictedBoltzmannMachine as DeepGBRBM,
        )

        self.assertIs(DeepGBRBM, GaussianBernoulliRestrictedBoltzmannMachine)


class TestPackageMetadata(unittest.TestCase):
    """Guard package metadata consumed by the build system."""

    def test_version_is_version_string(self):
        """__version__ feeds the pyproject dynamic version; it must be valid.

        ``[tool.setuptools.dynamic]`` in pyproject.toml reads
        ``kaiwu.torch_plugin.__version__``, so a malformed value breaks
        packaging rather than the test suite.
        """
        version = torch_plugin.__version__
        self.assertIsInstance(version, str)
        parts = version.split(".")
        self.assertGreaterEqual(len(parts), 2, f"version {version!r} is too short")
        for part in parts[:2]:
            self.assertTrue(
                part.isdigit(),
                f"version {version!r} must start with numeric release parts",
            )


if __name__ == "__main__":
    unittest.main()
