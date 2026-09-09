r"""The model boundary. Application code imports THIS — never openai/anthropic.

Implementations live beside this file: :class:`~retail_support.llm.openai_compat.OpenAICompatClient`
(OpenAI cloud, the course gateway, and vLLM — all three speak the same wire dialect),
:class:`~retail_support.llm.anthropic_client.AnthropicClient`, and
:class:`~retail_support.llm.fake.FakeClient` (tests and the eval harness).

The CI check that keeps this honest::

    grep -rn "import openai\|import anthropic" src/ --include="*.py"

must only ever match files inside ``src/retail_support/llm/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

FinishReason = Literal["stop", "length", "tool_calls", "refusal", "error"]


class ToolCall(BaseModel):
    """A tool the model wants the *application* to run. The model never runs it."""

    id: str
    name: str
    arguments: str  # raw JSON string; parsed and validated in Module 3


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # tool plumbing (Module 3) — absent for plain chat
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMRequest(BaseModel):
    messages: list[Message]
    model_alias: str = "retail_support-default"  # resolved via config, never a literal model id
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, gt=0)  # ALWAYS bounded
    tools: list[dict] | None = None  # provider-neutral JSON schema
    response_format: dict | None = None  # structured outputs (Module 3)
    cache_prefix_messages: int = 0  # how many leading messages are the stable prefix (M6)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # Module 6 pays close attention to this one


class LLMResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = "stop"
    model_id: str = ""  # the CONCRETE model that answered, never the alias
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    route: str = ""  # which configured route served it (set by the client)


class StreamChunk(BaseModel):
    """One frame of a streamed response. The final frame carries usage — losing it
    is the classic streaming cost-accounting bug (Module 2 §3)."""

    delta: str = ""
    final: bool = False
    usage: Usage | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    model_id: str = ""
    finish_reason: FinishReason | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Everything the application is allowed to know about a model provider."""

    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]: ...


class LLMError(RuntimeError):
    """Normalised provider failure. ``retryable`` drives Module 1's retry policy."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        raw: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw = raw

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"LLMError(status={self.status}, retryable={self.retryable}, msg={self!s})"
