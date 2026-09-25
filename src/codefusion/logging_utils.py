"""Structured logging helpers.

Every module logs through :func:`get_logger`. Log records carry an optional
``extra={"data": {...}}`` payload which is rendered as JSON (``CODEFUSION_LOG_FORMAT=json``)
or as ``key=value`` pairs (default, human-friendly).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

_CONFIGURED = False


class _KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name}: {record.getMessage()}"
        data = getattr(record, "data", None)
        if data:
            kv = " ".join(f"{k}={_short(v)}" for k, v in data.items())
            base = f"{base} | {kv}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        data = getattr(record, "data", None)
        if data:
            payload.update({k: _jsonable(v) for k, v in data.items()})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    return s if len(s) <= 120 else s[:117] + "..."


def _jsonable(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = (level or os.environ.get("CODEFUSION_LOG_LEVEL", "INFO")).upper()
    fmt = (fmt or os.environ.get("CODEFUSION_LOG_FORMAT", "kv")).lower()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if fmt == "json" else _KeyValueFormatter())
    root = logging.getLogger("codefusion")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    if not name.startswith("codefusion"):
        name = f"codefusion.{name}"
    return logging.getLogger(name)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **data: Any) -> None:
    """Log ``msg`` with a structured payload."""
    logger.log(level, msg, extra={"data": data})
