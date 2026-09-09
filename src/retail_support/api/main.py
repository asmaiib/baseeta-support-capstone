"""The API layer — thin on purpose.

It parses, calls the pipeline, and serialises. No prompt text, no provider SDK, no
business rule: every one of those lives behind ``build_assistant``. That is what
makes the same pipeline testable from ``pytest``, drivable from the CLI, runnable
in the replay harness and servable over HTTP without four copies of the logic.

    uvicorn retail_support.api.main:app --port 8000     # or: docker compose up app
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from retail_support.app import build_assistant
from retail_support.domain.session import Session
from retail_support.observability import get_logger

log = get_logger(__name__)

app = FastAPI(title="RetailSupport", version="1.0.0", description="Bilingual customer-services assistant")

_assistant = None
#: In-process sessions. A real deployment puts these in Redis with a TTL; the
#: shape of what is stored is the same, and that is the part worth learning.
_sessions: dict[str, Session] = {}


def assistant():
    global _assistant
    if _assistant is None:
        _assistant = build_assistant(route=os.environ.get("RETAIL_SUPPORT_ROUTE") or None)
    return _assistant


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str = Field("default", max_length=64)
    customer_id: str = Field("customer-A", max_length=64)


class AskResponse(BaseModel):
    text: str
    intent: str
    language: str
    blocked: bool
    guard_category: str
    model_id: str
    route: str
    cache_tier: str
    tool_calls: list[dict[str, Any]]
    latency_ms: float
    cost_halalas: float
    trace_id: str


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/readyz")
def readyz() -> dict:
    """Ready means the model boundary answers, not that the process started."""
    from retail_support.llm.interfaces import LLMRequest, Message

    try:
        response = assistant().deps.faq_handler._client.complete(  # noqa: SLF001
            LLMRequest(
                messages=[Message(role="user", content="ping")],
                max_tokens=8,
                model_alias="retail_support-small",
            )
        )
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "error": type(exc).__name__}
    return {"ready": True, "model_id": response.model_id}


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    session = _sessions.setdefault(
        request.session_id, Session(customer_id=request.customer_id)
    )
    reply = assistant().ask(request.message, session)
    return AskResponse(
        text=reply.text,
        intent=reply.intent,
        language=reply.language,
        blocked=reply.blocked,
        guard_category=reply.guard_category,
        model_id=reply.model_id,
        route=reply.route,
        cache_tier=reply.cache_tier,
        tool_calls=reply.tool_calls,
        latency_ms=round(reply.latency_ms, 1),
        cost_halalas=reply.cost_halalas,
        trace_id=reply.trace_id,
    )


@app.delete("/v1/sessions/{session_id}")
def clear_session(session_id: str) -> dict:
    _sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.get("/v1/cost")
def cost() -> dict:
    """Where the money went, since this process started."""
    meter = getattr(assistant(), "meter", None)
    if meter is None:
        return {"metered": False}
    return {
        "metered": True,
        "calls": len(meter.records),
        "total_halalas": meter.total_halalas,
        "by_intent": meter.by("intent"),
        "by_model": meter.by("model_id"),
        "prompt_cache_share": round(meter.cached_input_share(), 4),
    }
