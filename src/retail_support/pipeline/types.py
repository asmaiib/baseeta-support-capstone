"""What comes out of the pipeline.

One typed object, filled in stage by stage, so that every downstream consumer —
the CLI, the API, the replay harness, the eval runner — reads the same fields.
The eval harness runs through the *real* pipeline and reads this; a harness with
its own simplified copy of the request path drifts, and then measures the copy.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Reply(BaseModel):
    text: str = ""
    intent: str = "faq"
    language: str = "en"
    route: str = ""
    model_id: str = ""
    prompt_version: str = ""
    cache_tier: str = ""  # "", "exact", "semantic"
    blocked: bool = False
    guard_layer: str = "none"
    guard_category: str = "ok"
    output_guard_category: str = "ok"
    tool_calls: list[dict] = Field(default_factory=list)
    tool_iterations: int = 0
    escalated: bool = False
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    cost_halalas: float = 0.0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    trace_id: str = ""
    degraded: bool = False
