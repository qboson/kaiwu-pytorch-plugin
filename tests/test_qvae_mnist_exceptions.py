"""Regression tests for message preservation in qvae_mnist exceptions.

``utils.exception.Error`` used to call ``super().__init__()`` without the
message, so every raised qvae_mnist exception rendered as an empty string
in tracebacks, logs, and ``except ... as exc: str(exc)`` handlers. The
message was only visible through the module logger.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/qvae_mnist"))
)

from utils.exception import (  # noqa: E402  pylint: disable=wrong-import-position
    ArgumentError,
    BuildError,
    Error,
    SizeError,
    TypeError as QvaeTypeError,
    ValueError as QvaeValueError,
)


@pytest.mark.parametrize(
    "exc_type, message",
    [
        (ArgumentError, "wrong number of provided arguments"),
        (BuildError, "model has not been built"),
        (SizeError, "wrong length of variables"),
        (QvaeTypeError, "wrong type of variables"),
        (QvaeValueError, "unsupported model type"),
    ],
)
def test_exception_subclasses_preserve_their_message(exc_type, message):
    """``str(exc)`` must expose the message passed to the constructor."""
    assert str(exc_type(message)) == message
    assert exc_type(message).args == (message,)


def test_base_error_still_logs_and_carries_the_message():
    """The generic ``Error`` class keeps its message as well."""
    error = Error("SomeClass", "something failed")
    assert str(error) == "something failed"


def test_custom_value_error_is_matchable_by_message():
    """``pytest.raises(..., match=...)`` and generic handlers see the text."""
    with pytest.raises(QvaeValueError, match="unsupported model type"):
        raise QvaeValueError("unsupported model type: CellQVAE")


def test_custom_type_error_still_satisfies_builtin_isinstance_checks():
    """Dual inheritance must keep working with the builtin counterparts."""
    error = QvaeTypeError("boom")
    assert isinstance(error, TypeError)
    assert str(error) == "boom"


def test_trainer_value_error_keeps_its_message():
    """The trainer module shadows ``ValueError`` with the custom class."""
    pytest.importorskip("gif")
    pytest.importorskip("imageio")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    from trainer import trainer as trainer_module  # noqa: E402  pylint: disable=wrong-import-position,import-outside-toplevel

    assert trainer_module.ValueError is QvaeValueError
    with pytest.raises(ValueError, match="Unsupported model type"):
        raise trainer_module.ValueError("Unsupported model type: CellQVAE")
