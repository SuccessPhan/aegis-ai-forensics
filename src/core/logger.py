"""Centralised logger for AEGIS. Use get_logger(__name__); never print()."""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("aegis")
    level = os.environ.get("AEGIS_LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FMT))
        root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "aegis") -> logging.Logger:
    """Return a child of the 'aegis' logger (level via env AEGIS_LOG_LEVEL)."""
    _configure_root()
    if name != "aegis" and not name.startswith("aegis."):
        name = f"aegis.{name}"
    return logging.getLogger(name)
