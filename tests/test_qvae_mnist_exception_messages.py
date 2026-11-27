"""Keep QVAE example exception messages visible without optional dependencies."""

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


EXAMPLE = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
PRELUDE = """
import builtins
import pickle
from pathlib import Path
import sys

example = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(example))
from utils import exception as errors
logging_helper = sys.modules[errors.get_logger.__module__]

assert Path(errors.__file__).resolve() == example / "utils" / "exception.py"
assert Path(logging_helper.__file__).resolve().parent == example / "utils"
concrete = (
    errors.ArgumentError, errors.BuildError, errors.SizeError,
    errors.TypeError, errors.ValueError,
)
"""


class TestExceptionMessages(unittest.TestCase):
    """Use fresh normal imports and real log handlers in a disposable directory."""

    def run_case(self, code):
        # Isolate the example's generic `utils` namespace and relative log file.
        with tempfile.TemporaryDirectory(prefix="qvae-exceptions-") as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c",
                 PRELUDE + textwrap.dedent(code), str(EXAMPLE)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_positional_messages_and_args(self):
        self.run_case("""
            for message in ("invalid configuration", ""):
                instances = [errors.Error("Model", message)]
                instances.extend(cls(message) for cls in concrete)
                for error in instances:
                    assert str(error) == message, (type(error), str(error), message)
                    assert error.args == (message,), (type(error), error.args)
        """)

    def test_keyword_messages_and_args(self):
        self.run_case("""
            for message in ("invalid configuration", ""):
                instances = [errors.Error(cls="Model", msg=message)]
                instances.extend(cls(error=message) for cls in concrete)
                for error in instances:
                    assert str(error) == message, (type(error), str(error), message)
                    assert error.args == (message,), (type(error), error.args)
        """)

    def test_concrete_exceptions_survive_pickle(self):
        self.run_case("""
            for cls in concrete:
                for message in ("invalid configuration", ""):
                    original = cls(error=message)
                    restored = pickle.loads(pickle.dumps(original))
                    assert type(restored) is cls
                    assert restored.args == (message,), restored.args
                    assert str(restored) == message
        """)

    def test_builtin_exception_compatibility(self):
        self.run_case("""
            for cls, builtin in (
                (errors.TypeError, builtins.TypeError),
                (errors.ValueError, builtins.ValueError),
            ):
                error = cls("invalid configuration")
                assert isinstance(error, builtin)
                try:
                    raise error
                except builtin as caught:
                    assert caught is error
        """)

    def test_real_logging_keeps_class_markers_and_messages(self):
        result = self.run_case("""
            errors.Error("Model", "base detail")
            for cls in concrete:
                cls("concrete detail")
            for handler in errors.logger.handlers:
                handler.flush()
            log = Path("training.log").read_text()
            assert "Model: base detail." in log
            for cls in concrete:
                assert f"{cls.__name__}: concrete detail." in log
        """)
        self.assertIn("Model: base detail.", result.stdout)
        for name in ("ArgumentError", "BuildError", "SizeError", "TypeError", "ValueError"):
            self.assertIn(f"{name}: concrete detail.", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
