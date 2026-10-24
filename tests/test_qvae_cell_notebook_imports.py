"""Notebook import-resolution contract for the single-cell example."""

import ast
import importlib
import json
import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLE_DIR = os.path.join(REPO_ROOT, "example", "qvae_cell")
LOCAL_ROOTS = {"trainer", "visualization", "evaluate_benchmark"}


def prepare_environment():
    """Installs the import stubs this example needs (optional dependencies)."""
    for name in ("scanpy", "anndata"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    import kaiwu.preprocess as preprocess

    if not hasattr(preprocess, "PrecisionReducer"):
        preprocess.PrecisionReducer = type(
            "PrecisionReducer", (), {"__init__": lambda self, *args, **kwargs: None}
        )
    for name in [m for m in list(sys.modules) if m == "models" or m.startswith("models.")]:
        del sys.modules[name]
    for name in [m for m in list(sys.modules) if m == "trainer" or m.startswith("trainer.")]:
        del sys.modules[name]
    sys.path.insert(0, EXAMPLE_DIR)


def notebook_cells():
    """Yields the source of every code cell in the notebook."""
    path = os.path.join(EXAMPLE_DIR, "train_qvae_cell.ipynb")
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
    prepare_environment()
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
