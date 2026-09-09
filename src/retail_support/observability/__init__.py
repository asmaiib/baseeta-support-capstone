"""Structured logging. Every model call gets a trace id, model id, latency, token
usage and finish reason — the substrate Module 6 monetises and SDA-AIE-312 turns
into observability.

One JSON object per line, so ``jq`` is the analytics tool (Module 6's party trick)::

    jq -s 'group_by(.intent) | map({intent: .[0].intent,
           halalas: (map(.cost_halalas) | add)})' logs/llm_cost.jsonl
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path

import structlog

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_configured = False


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def current_trace_id() -> str:
    return _trace_id.get()


def _add_trace_id(_logger, _name, event_dict):
    if tid := _trace_id.get():
        event_dict.setdefault("trace_id", tid)
    return event_dict


def configure_logging(*, json_lines: bool | None = None, path: str | Path | None = None) -> None:
    """Console-friendly by default; JSON lines when ``RETAIL_SUPPORT_LOG_JSON=1`` or a path
    is given. The walkthroughs read the JSON form with ``jq``; humans read the other one."""
    global _configured
    json_lines = (
        json_lines if json_lines is not None else os.environ.get("RETAIL_SUPPORT_LOG_JSON") == "1"
    )
    path = path or os.environ.get("RETAIL_SUPPORT_LOG_FILE")

    stream = sys.stderr
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        stream = open(path, "a", encoding="utf-8")  # noqa: SIM115 - process-lifetime handle
        json_lines = True

    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_lines
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_trace_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, os.environ.get("RETAIL_SUPPORT_LOG_LEVEL", "INFO").upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "retail_support"):
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
