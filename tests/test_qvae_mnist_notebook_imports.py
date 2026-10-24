"""Notebook import-resolution contract for the QVAE-MNIST example."""

import ast
import importlib
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLE_DIR = os.path.join(REPO_ROOT, "example", "qvae_mnist")
sys.path.insert(0, EXAMPLE_DIR)

for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]

LOCAL_ROOTS = {"model", "utils", "downstream", "trainer"}


def notebook_cells(name):
    """Yields concatenated source for every code cell of a notebook."""
    path = os.path.join(EXAMPLE_DIR, name)
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            yield "".join(cell.get("source", []))


def local_imports(source):
    """Returns local module names imported by one cell."""
    names = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("from ") or stripped.startswith("import "):
            try:
                tree = ast.parse(stripped)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module:
                    root = node.module.split(".")[0]
                    if root in LOCAL_ROOTS:
                        names.append(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root in LOCAL_ROOTS:
                            names.append(alias.name)
    return names


def import_with_qvae_utils(module):
    """Imports one local module with this example's ``utils`` winning the lookup.

    Other example tests put their own top-level ``utils``/``workflows`` packages
    on ``sys.path``; they are hidden for the duration of the import and restored
    afterwards so the rest of the session is unaffected.
    """
    hidden = {
        name: value
        for name, value in list(sys.modules.items())
        if name in {"utils", "workflows"}
        or name.startswith("utils.")
        or name.startswith("workflows.")
    }
    for name in hidden:
        del sys.modules[name]
    original_path = list(sys.path)
    sys.path[:] = [entry for entry in original_path if not entry.endswith("dplm")]
    sys.path.insert(0, EXAMPLE_DIR)
    try:
        return importlib.import_module(module)
    finally:
        sys.path[:] = original_path
        sys.modules.update(hidden)


def test_every_local_notebook_import_resolves():
    """Each local module imported by the notebooks can be imported."""
    modules = set()
    for name in ("train_qvae.ipynb", "train_qvae_classifier.ipynb"):
        for source in notebook_cells(name):
            modules.update(local_imports(source))
    assert modules, "no local imports found in the notebooks"
    for module in sorted(modules):
        import_with_qvae_utils(module)


def test_run_magics_point_at_existing_files():
    """Every ``%run`` target exists next to the notebook."""
    targets = []
    for name in ("train_qvae.ipynb", "train_qvae_classifier.ipynb"):
        for source in notebook_cells(name):
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("%run "):
                    targets.append(stripped.split(None, 1)[1].split()[0])
    for target in targets:
        assert os.path.exists(os.path.join(EXAMPLE_DIR, target)), target
