from __future__ import annotations

import logging
from logging import Logger
from typing import Optional


LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARN,
    "error": logging.ERROR,
}


def setup_logging(debug: bool = False, quiet: bool = False, log_file: Optional[str] = None) -> Logger:
    level = logging.DEBUG if debug else (logging.WARN if quiet else logging.INFO)
    logger = logging.getLogger("adp")
    logger.setLevel(level)
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    logger.debug("Logger initialized (debug=%s, quiet=%s, file=%s)", debug, quiet, log_file)
    return logger
