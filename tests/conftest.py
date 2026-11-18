"""Test bootstrap: make this working tree, not the installed copy, the import target.

Why this file exists
--------------------
``kaiwu`` is installed into the environment with ``pip install .``.  When the test
suite starts from a repository checkout, ``import kaiwu.torch_plugin`` resolves to
the *installed* copy under ``site-packages`` instead of the code under ``src/``.
A local ``pytest tests/`` run therefore validates the last installed build rather
than the working tree: edits under ``src/`` are invisible to it, and the suite can
stay green while the code it is supposed to guard is broken.

This file is a local development affordance and is scoped to be one: it is enabled
by default *outside* continuous integration and disabled *inside* it, keyed on the
``CI`` environment variable every major CI provider exports.  Locally the checkout
is what you are editing, so it is what has to be imported.  In CI the ``pytest``
step has the opposite job -- it validates the artifact that ``pip install .``
produced, which is why that workflow re-runs the suite against the installed tree
and asserts the resolution separately with ``python -I -c ...``; re-ranking the
package path there would quietly downgrade artifact validation to a checkout run.
Where the two trees happen to be byte-identical -- as ``src/kaiwu`` and the
installed ``kaiwu`` are on an ordinary revision -- the outcomes coincide anyway,
but that is a consequence of the current revision, not what makes the behaviour
safe.  Set ``CI`` or ``KAIWU_TESTS_KEEP_INSTALLED`` to exercise the installed copy
on purpose.

The correct sequence is not a discovery: ``tests/test_qvae_dataset_mean_state.py``
already performs it, at import time, for its own imports.  It lives in a single test
module, so the rest of the suite keeps importing the installed copy, and the outcome
of a full ``pytest tests/`` run ends up depending on collection order.  Repeating that
block in every test module would be worse; this file makes the behaviour the default
for ``pytest``, which is the only reason it lives in ``tests/``.

Four bootstrap patterns that look right and are not
---------------------------------------------------
1. Guarded prepend, e.g. ``if src not in kaiwu.__path__: kaiwu.__path__ = [src] + ...``
   The guard is already false: ``kaiwu/__init__.py`` calls ``pkgutil.extend_path``,
   which appends every ``kaiwu`` directory found on ``sys.path`` -- ``src/kaiwu``
   included -- to ``__path__``.  The entry is present but ranked *last*, so the
   prepend never happens.
2. Appending instead of prepending, e.g. ``kaiwu.__path__ = list(kaiwu.__path__) + [src]``
   ``extend_path`` keeps the installed regular package first, and a path appended
   after it is only consulted once that one is exhausted, i.e. never for modules
   that exist in both trees.
3. Prepending without clearing ``sys.modules``
   ``kaiwu/__init__.py`` eagerly imports ``kaiwu.torch_plugin``, so the installed
   submodule object is already cached.  Reordering ``__path__`` afterwards does not
   invalidate that cache entry and ``import kaiwu.torch_plugin`` keeps returning the
   installed module; the package attribute left behind by that eager import has to
   be dropped as well.
4. Only touching ``sys.path``
   ``src/kaiwu`` ships no ``__init__.py`` -- it is an *extension directory* holding
   nothing but ``torch_plugin/``, contributed to the ``kaiwu`` package installed from
   the wheel.  It is therefore a PEP 420 namespace portion and always loses against
   the installed regular package, no matter where it sits on ``sys.path``.  Submodule
   lookups go through ``kaiwu.__path__``, so that is what has to be reordered;
   ``sys.path`` is deliberately left untouched here.

``__path__``, ``sys.modules`` and the package attribute therefore have to be fixed
together, which is what ``_prefer_working_tree()`` does below.  The helper is
deliberately conservative: it touches nothing but the ``kaiwu.torch_plugin``
subtree, is a no-op from the second call onwards, and silently does nothing when
the checkout layout is unexpected (installed-only tree, missing ``kaiwu``, or an
environment that already resolves to this tree).  It also does nothing when
``CI`` is set, so CI keeps validating the installed artifact.  Set
``KAIWU_TESTS_KEEP_INSTALLED=1`` to switch the behaviour off locally and exercise
the installed copy on purpose.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_PLUGIN = "kaiwu.torch_plugin"
_SRC_KAIWU = Path(__file__).resolve().parents[1] / "src" / "kaiwu"
_OPT_OUT_ENV = "KAIWU_TESTS_KEEP_INSTALLED"
_CI_ENV = "CI"


def _same_path(left: object, right: object) -> bool:
    """Compare two filesystem paths, tolerating case, symlinks and separators."""
    try:
        return os.path.normcase(os.path.realpath(os.fspath(left))) == os.path.normcase(
            os.path.realpath(os.fspath(right))
        )
    except TypeError:  # pragma: no cover - defensive: ``__path__`` entries are strings
        return False


def _is_kaiwu_tree(candidate: Path) -> bool:
    """Tell whether ``candidate`` can provide ``kaiwu.torch_plugin``.

    ``src/kaiwu`` is an extension directory: no ``__init__.py``, only
    ``torch_plugin/``.  Requiring ``__init__.py`` here would silently disable this
    whole file -- exactly the failure mode it exists to remove -- so the real
    criterion is whether the ``torch_plugin`` subtree is present.
    """
    if not candidate.is_dir():
        return False
    if (candidate / "__init__.py").is_file():
        return True
    return (candidate / "torch_plugin").is_dir()


def _module_origin(module_name: str) -> str | None:
    """Return ``__file__`` of an already imported module, or ``None``."""
    return getattr(sys.modules.get(module_name), "__file__", None)


def _drop_cached_plugin_modules() -> None:
    """Drop ``kaiwu.torch_plugin`` and its submodules from ``sys.modules``.

    Only that subtree: ``kaiwu`` itself and the sibling subpackages its
    ``__init__.py`` eagerly imports (``kaiwu.core``, ``kaiwu.classical``,
    ``kaiwu.cim`` ...) are left alone, because they are provided by the installed
    package and are not rebuilt once deleted.
    """
    dropped = {
        name: sys.modules.pop(name)
        for name in [key for key in sys.modules if key == _PLUGIN or key.startswith(_PLUGIN + ".")]
    }
    package = sys.modules.get("kaiwu")
    if package is None:
        return
    for name, module in dropped.items():
        attribute = name[len("kaiwu.") :]
        if "." in attribute:
            continue
        if getattr(package, attribute, None) is module:
            delattr(package, attribute)


def _rank_first(package: object, entries: list[str]) -> None:
    """Reorder the package search path in place, keeping trailing entries intact."""
    search_path = package.__path__  # type: ignore[attr-defined]
    try:
        search_path[:] = entries  # keeps list identity, so ``__spec__`` stays in sync
    except TypeError:  # pragma: no cover - namespace packages expose a read-only path
        package.__path__ = entries  # type: ignore[attr-defined]
    spec = getattr(package, "__spec__", None)
    locations = getattr(spec, "submodule_search_locations", None)
    if locations is not None and locations is not package.__path__:  # type: ignore[attr-defined]
        try:
            locations[:] = entries
        except TypeError:  # pragma: no cover - defensive
            pass


def _prefer_working_tree() -> bool:
    """Rank ``src/kaiwu`` first for ``kaiwu.torch_plugin`` imports.

    Returns ``True`` when the import state was changed and ``False`` when the call
    was a no-op.  Anything unexpected is treated as "nothing to do" rather than an
    error, so the suite still runs against whatever installation is present.
    """
    if os.environ.get(_OPT_OUT_ENV):
        return False
    # CI validates the artifact built by ``pip install .``; never re-rank there.
    if os.environ.get(_CI_ENV):
        return False
    if not _is_kaiwu_tree(_SRC_KAIWU):
        return False
    try:
        # Imported here so that an environment without the package degrades quietly.
        import kaiwu  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False

    working_tree_plugin = _SRC_KAIWU / "torch_plugin"
    origin = _module_origin(_PLUGIN)
    ranked_first = bool(kaiwu.__path__) and _same_path(kaiwu.__path__[0], _SRC_KAIWU)
    origin_matches = origin is None or _same_path(
        os.path.dirname(origin), working_tree_plugin
    )
    if ranked_first and origin_matches:
        return False  # already done: repeated imports and later sessions are no-ops

    _drop_cached_plugin_modules()
    entries = [os.fspath(_SRC_KAIWU)]
    entries.extend(entry for entry in list(kaiwu.__path__) if not _same_path(entry, _SRC_KAIWU))
    _rank_first(kaiwu, entries)
    importlib.invalidate_caches()
    return True


_prefer_working_tree()
