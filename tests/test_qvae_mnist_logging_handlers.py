"""Logging-handler contract for the QVAE-MNIST example."""

import importlib.util
import logging
import os
from logging.handlers import TimedRotatingFileHandler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_PATH = os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "logging.py")

_spec = importlib.util.spec_from_file_location("qvae_mnist_logging_handlers", MODULE_PATH)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


def test_console_handler_streams_to_stdout_with_the_shared_formatter():
    """The console handler is a stdout StreamHandler using FORMATTER."""
    handler = module.get_console_handler()
    assert isinstance(handler, logging.StreamHandler)
    assert handler.formatter._fmt == module.FORMATTER._fmt
    assert "%(levelname)s" in module.FORMATTER._fmt


def test_file_handler_rotates_at_midnight_and_defers_file_creation():
    """The file handler rotates daily, writes training.log and is lazy."""
    handler = module.get_file_handler()
    assert isinstance(handler, TimedRotatingFileHandler)
    assert handler.when == "MIDNIGHT"
    assert handler.delay is True
    assert handler.baseFilename.endswith(os.path.join("training.log"))
    assert handler.formatter._fmt == module.FORMATTER._fmt
    handler.close()
