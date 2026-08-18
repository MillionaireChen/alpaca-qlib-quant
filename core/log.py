"""Logging setup: console + timestamped file under logs/."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from core.config import REPO_ROOT

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, log_dir: str | Path = "logs") -> logging.Logger:
    """Return a logger writing to console and to ``logs/<name>_<timestamp>.log``."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured in this process
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(_FORMAT)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    ldir = Path(log_dir)
    if not ldir.is_absolute():
        ldir = REPO_ROOT / ldir
    ldir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    fh = logging.FileHandler(ldir / f"{name}_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
