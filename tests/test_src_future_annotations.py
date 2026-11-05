"""Every ``src`` module using modern annotation syntax must defer annotations.

``pyproject.toml`` declares ``requires-python = ">=3.8"``. Annotations such as
``torch.Tensor | None`` or ``list[int]`` are only safe on 3.8/3.9 when the module
imports ``annotations`` from ``__future__`` (otherwise they are evaluated at
definition time and raise ``TypeError``), so a module that forgets the import
breaks the package on the supported interpreters.
"""

import ast
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src", "kaiwu")
BUILTIN_GENERICS = {"list", "dict", "set", "tuple", "frozenset", "type"}


def deferred_annotations(path):
    """Returns whether the module opts into deferred annotations."""
    with open(path, encoding="utf-8") as handle:
        return "from __future__ import annotations" in handle.read()


def annotation_offenders(path):
    """Returns line numbers with PEP 604 / builtin-generic annotations."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    found = []
    for node in ast.walk(tree):
        annotations = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = list(node.args.posonlyargs) + list(node.args.args)
            arguments += list(node.args.kwonlyargs)
            annotations = [arg.annotation for arg in arguments if arg.annotation is not None]
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            annotations = [node.annotation]
        for annotation in annotations:
            for sub in ast.walk(annotation):
                modern_union = isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr)
                builtin_generic = (
                    isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id in BUILTIN_GENERICS
                )
                if modern_union or builtin_generic:
                    found.append(node.lineno)
                    break
            else:
                continue
            break
    return found


def test_modern_annotations_are_always_deferred():
    """No module without the future import may use 3.10-only annotation syntax."""
    scanned = 0
    offenders = []
    for root, _dirs, files in os.walk(SRC_ROOT):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            scanned += 1
            if deferred_annotations(path):
                continue
            for line in annotation_offenders(path):
                offenders.append(f"{os.path.relpath(path, REPO_ROOT)}:{line}")
    assert scanned > 0, "src tree was not found"
    assert offenders == []
