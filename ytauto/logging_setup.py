from __future__ import annotations

import logging
import logging.handlers
import sys
from typing import Optional

from .config import get_settings, log_dir

_CONFIGURED = False


class _Fmt(logging.Formatter):
    default_time_format = "%Y-%m-%d %H:%M:%S"

    def __init__(self, color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-22s %(message)s")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        out = super().format(record)
        if not self.color:
            return out
        colors = {
            "DEBUG": "\033[2;37m",
            "INFO": "\033[0;36m",
            "WARNING": "\033[0;33m",
            "ERROR": "\033[1;31m",
            "CRITICAL": "\033[1;35m",
        }
        return f"{colors.get(record.levelname, '')}{out}\033[0m"


def setup_logging(level: Optional[str] = None, quiet: bool = False) -> None:
    global _CONFIGURED
    root = logging.getLogger("ytauto")
    if _CONFIGURED:
        if level:
            root.setLevel(level.upper())
        return

    root.setLevel((level or "INFO").upper())
    root.propagate = False

    color = sys.stderr.isatty()
    fmt = _Fmt(color=color)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    if quiet:
        stream.setLevel(logging.WARNING)
    root.addHandler(stream)

    log_dir().mkdir(parents=True, exist_ok=True)
    rotating = logging.handlers.TimedRotatingFileHandler(
        log_dir() / "ytauto.log", when="midnight", backupCount=30, encoding="utf-8"
    )
    rotating.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(rotating)

    # googleapiclient is extremely chatty at INFO.
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    settings = get_settings()
    setup_logging()
    del settings
    return logging.getLogger(f"ytauto.{name}")
