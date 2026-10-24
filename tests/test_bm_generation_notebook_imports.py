"""Notebook import-resolution contract for the BM generation example."""

import ast
import importlib
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLE_DIR = os.path.join(REPO_ROOT, "example", "bm_generation")
LOCAL_ROOTS = {"data_loader", "saver", "trainer"}


def notebook_cells():
    """Yields the source of every code cell in the training notebook."""
    path = os.path.join(EXAMPLE_DIR, "train_bm.ipynb")
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
        if not stripped.startswith(("from ", "import ")):
            continue
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in LOCAL_ROOTS:
                    names.append((node.module, [alias.name for alias in node.names]))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in LOCAL_ROOTS:
                        names.append((alias.name, []))
    return names


def test_notebook_local_imports_resolve():
    """Every local module (and imported name) referenced by the notebook imports."""
    sys.path.insert(0, EXAMPLE_DIR)
    imports = []
    for source in notebook_cells():
        imports.extend(local_imports(source))
    assert imports, "no local imports found in the notebook"
    for module, names in imports:
        imported = importlib.import_module(module)
        for name in names:
            assert hasattr(imported, name), f"{module}.{name}"


def test_run_magics_point_at_existing_files():
    """Every ``%run`` target exists next to the notebook."""
    for source in notebook_cells():
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("%run "):
                target = stripped.split(None, 1)[1].split()[0]
                assert os.path.exists(os.path.join(EXAMPLE_DIR, target)), target
