"""Regression: docs/conf.py must resolve ``src/`` from its own file, not CWD.

Before the fix, ``os.path.abspath("../src")`` resolved against the process
working directory. A Sphinx build launched from the repository root (as Read
the Docs and most CI invocations do) therefore inserted a non-existent
directory at ``sys.path[0]``; when kaiwu-pytorch-plugin was not already
installed, the subsequent ``from kaiwu.torch_plugin import __version__`` failed
with ``ModuleNotFoundError`` and the docs build aborted.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "docs" / "conf.py"
SRC = REPO / "src"

# conf.py imports sphinx and the (torch-dependent) kaiwu package at module
# level. Stub both so the probe needs only the standard library.
_PROBE = r'''
import sys
import types

sphinx = types.ModuleType("sphinx")
sphinx.addnodes = types.ModuleType("addnodes")
sys.modules["sphinx"] = sphinx

kaiwu = types.ModuleType("kaiwu")
kaiwu.__path__ = []
torch_plugin = types.ModuleType("kaiwu.torch_plugin")
torch_plugin.__version__ = "0.0.0"
sys.modules["kaiwu"] = kaiwu
sys.modules["kaiwu.torch_plugin"] = torch_plugin

conf_path = sys.argv[1]
namespace = {"__file__": conf_path, "__name__": "conf_probe"}
with open(conf_path, encoding="utf-8") as handle:
    exec(compile(handle.read(), conf_path, "exec"), namespace)
print(sys.path[0])
'''


def _inserted_path(cwd):
    """Return sys.path[0] after conf.py runs with `cwd` as working directory."""
    result = subprocess.check_output(
        [sys.executable, "-c", _PROBE, str(CONF)],
        cwd=str(cwd),
        text=True,
    )
    return Path(result.strip())


def test_conf_src_path_is_independent_of_working_directory():
    """conf.py must insert <repo>/src regardless of where Sphinx was started."""
    assert _inserted_path(REPO) == SRC
    assert _inserted_path(REPO / "docs") == SRC
    assert _inserted_path(REPO.parent) == SRC


def test_conf_src_path_points_at_a_real_directory():
    """The inserted path must exist (it was a bogus <parent>/src before)."""
    assert _inserted_path(REPO.parent).is_dir()


def test_building_from_docs_directory_is_unchanged():
    """Legal input: the documented `make -C docs html` case is identical."""
    assert _inserted_path(REPO / "docs") == SRC
