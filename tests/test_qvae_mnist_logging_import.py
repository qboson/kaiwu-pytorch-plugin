"""Regression coverage for the MNIST example's logging module name."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


EXAMPLE = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"


class LoggingImportTests(unittest.TestCase):
    """Exercise imports in fresh interpreters without loading a model."""

    def run_python(self, source, cwd, *arguments):
        """Run with the real working directory on Python's search path."""
        environment = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source), *map(str, arguments)],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_stdlib_logging_import_from_utils(self):
        """The example directory must not shadow the standard library."""
        self.run_python(
            """
            import logging
            import logging.handlers
            from pathlib import Path
            import sysconfig

            assert Path(logging.__file__).resolve().is_relative_to(
                Path(sysconfig.get_path("stdlib")).resolve()
            )
            assert logging.handlers.TimedRotatingFileHandler
            """,
            EXAMPLE / "utils",
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_torch_import_from_utils(self):
        """Use the real PyTorch import that originally exposed the collision."""
        self.run_python(
            """
            import torch
            import sys

            assert torch.__version__
            assert "kaiwu" not in sys.modules
            """,
            EXAMPLE / "utils",
        )

    def test_logger_behavior_and_exception_import(self):
        """Keep real handlers, idempotence, output and exception imports working."""
        with tempfile.TemporaryDirectory() as directory:
            self.run_python(
                """
                import sys
                sys.path.insert(0, sys.argv[1])
                from utils.kpp_logging import get_logger
                from utils import exception
                import logging
                from logging.handlers import TimedRotatingFileHandler
                from pathlib import Path

                logger = get_logger("kpp.logging.import.regression")
                assert get_logger(logger.name) is logger
                assert len(logger.handlers) == 2
                console, file_handler = logger.handlers
                assert type(console) is logging.StreamHandler
                assert type(file_handler) is TimedRotatingFileHandler
                assert console.stream is sys.stdout
                assert file_handler.when == "MIDNIGHT"
                assert file_handler.delay is True
                assert logger.level == logging.DEBUG
                assert logger.propagate is False
                assert not Path("training.log").exists()
                logger.info("real logging regression message")
                file_handler.flush()
                assert "real logging regression message" in Path(
                    "training.log"
                ).read_text()
                assert issubclass(exception.ValueError, ValueError)
                assert "kaiwu" not in sys.modules
                """,
                directory,
                EXAMPLE,
            )

    def test_consumer_logging_import_statements(self):
        """Resolve the actual imports without executing trainer/model code."""
        with tempfile.TemporaryDirectory() as directory:
            self.run_python(
                """
                import ast
                from pathlib import Path
                import sys

                example = Path(sys.argv[1])
                sys.path.insert(0, str(example))
                for relative, package in (
                    ("trainer/trainer.py", "trainer"),
                    ("trainer/model_tuner.py", "trainer"),
                    ("utils/exception.py", "utils"),
                ):
                    path = example / relative
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                    imports = [
                        node for node in tree.body
                        if isinstance(node, ast.ImportFrom)
                        and any(alias.name == "get_logger" for alias in node.names)
                    ]
                    assert len(imports) == 1, relative
                    namespace = {"__package__": package}
                    statement = ast.Module(body=imports, type_ignores=[])
                    exec(compile(statement, str(path), "exec"), namespace)
                    assert callable(namespace["get_logger"])
                assert "kaiwu" not in sys.modules
                """,
                directory,
                EXAMPLE,
            )


if __name__ == "__main__":
    unittest.main()
