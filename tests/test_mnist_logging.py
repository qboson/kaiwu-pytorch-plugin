"""Regression checks for portable MNIST training log files."""

import importlib.util
import io
import logging
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "example/qvae_mnist/utils/logging.py"
)
SPEC = importlib.util.spec_from_file_location("mnist_logging_utils", MODULE_PATH)
LOGGING_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOGGING_UTILS)


@pytest.mark.parametrize("default_encoding", ["utf-8", "cp1252"])
def test_unicode_log_messages_survive_rotation(tmp_path, monkeypatch, default_encoding):
    """Write both sides of a rollover under UTF-8 and legacy locale defaults."""
    log_path = tmp_path / "training.log"
    monkeypatch.setattr(LOGGING_UTILS, "LOG_FILE", str(log_path))
    # FileHandler resolves an omitted encoding through this standard-library
    # function. Emulate the locale here, then exercise real file I/O below.
    with monkeypatch.context() as encoding_context:
        encoding_context.setattr(
            io,
            "text_encoding",
            lambda encoding, stacklevel=2: default_encoding
            if encoding is None
            else encoding,
        )
        handler = LOGGING_UTILS.get_file_handler()

    before = "Before rotation: 训练 🚀"
    after = "After rotation: δοκιμή 🧪"
    try:
        handler.emit(
            logging.LogRecord("training", logging.INFO, __file__, 0, before, (), None)
        )
        handler.doRollover()
        handler.emit(
            logging.LogRecord("training", logging.INFO, __file__, 0, after, (), None)
        )
    finally:
        handler.close()

    rotated_files = list(tmp_path.glob("training.log.*"))
    assert len(rotated_files) == 1
    assert before in rotated_files[0].read_text(encoding="utf-8")
    assert after in log_path.read_text(encoding="utf-8")
