"""Logger-factory contract for the QVAE-MNIST example.

The module is loaded by path: several example trees ship their own ``utils``
package, so importing it as ``utils.logging`` can resolve to another example
when the whole test suite runs in one process.
"""

import importlib.util
import logging
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_PATH = os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "logging.py")

_spec = importlib.util.spec_from_file_location("qvae_mnist_logging", MODULE_PATH)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)

get_logger = module.get_logger


def reset_logger(name):
    """Clears handlers so each test starts from an unconfigured logger."""
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    return logger


def test_logger_is_configured_once_with_console_and_file_handlers():
    """The first call sets DEBUG level, two handlers and disables propagation."""
    logger = reset_logger("qvae_contract_logger")
    configured = get_logger("qvae_contract_logger")
    assert configured is logger
    assert configured.level == logging.DEBUG
    assert len(configured.handlers) == 2
    assert isinstance(configured.handlers[0], logging.StreamHandler)
    assert configured.propagate is False


def test_repeated_calls_do_not_add_handlers():
    """Calling the factory twice keeps the handler list unchanged."""
    logger = reset_logger("qvae_contract_logger_twice")
    first = get_logger("qvae_contract_logger_twice")
    handler_count = len(first.handlers)
    second = get_logger("qvae_contract_logger_twice")
    assert second is first
    assert len(second.handlers) == handler_count == 2
