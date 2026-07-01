# -*- coding: utf-8 -*-
"""
Logging-based methods and helpers.
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "training.log"

def get_console_handler():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(FORMATTER)
    return handler

def get_file_handler():
    handler = TimedRotatingFileHandler(LOG_FILE, when="midnight", delay=True)
    handler.setFormatter(FORMATTER)
    return handler

def get_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler())
        logger.propagate = False
    return logger