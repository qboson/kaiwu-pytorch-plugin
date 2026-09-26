# -*- coding: utf-8 -*-
"""Logging helpers: get a named logger, optionally writing to a file."""
import logging
import sys
from pathlib import Path

_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s — %(levelname)s — %(message)s"
)


def get_logger(name: str, log_file=None):
    """Return a logger. If log_file is given, ensure a file handler is attached."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                      for h in logger.handlers)
    if not has_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(_FORMATTER)
        logger.addHandler(ch)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        already = any(getattr(h, "baseFilename", None) == str(Path(log_file).resolve())
                      for h in logger.handlers if isinstance(h, logging.FileHandler))
        if not already:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(_FORMATTER)
            logger.addHandler(fh)
    return logger
