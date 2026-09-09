"""The validate -> retry -> repair loop, once, for everything structured.

Guards are extraction. Routing is extraction. Judging is extraction. They all use
this function, which is why Module 4 and Module 5 cost almost no new machinery —
they reuse Module 3's.

The loop, and why each step is there:

1. **Parse** — guaranteed by JSON mode and above, so parsing is not where you
   spend your error budget.
2. **Validate** with pydantic. On success a *typed object* crosses into the
   application and nothing downstream ever touches raw model output.
3. **On failure, retry once with the validation errors rendered verbatim.** Models
   repair reliably against explicit, located errors; "please try again" barely
   moves the pass rate.
4. **On the second failure, escalate by design.** Never loop unboundedly — each
   retry is money and latency — and never relax the schema in the heat of an
   incident. Failures belong on the escalation path, not in the contract.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from retail_support.llm.interfaces import LLMClient, LLMRequest, LLMResponse, Message
from retail_support.observability import get_logger

log = get_logger(__name__)

#: ruff would rather this were a PEP 695 type parameter. It is a TypeVar because
#: the signature reads better in a teaching codebase, and because the generic is
#: the point of the function: one loop, any contract.
T = TypeVar("T", bound=BaseModel)

REPAIR_INSTRUCTION = (
    "The JSON you returned failed validation. Fix ONLY these errors and return the "
    "corrected object, with no commentary:\n{errors}"
)


class StructuredExtractionFailed(Exception):
    """Carries the raw output and the errors onward — to the human-review queue."""

    def __init__(self, raw: str | None, errors: list[dict], attempts: int) -> None:
        super().__init__(f"structured extraction failed after {attempts} attempts")
        self.raw = raw
        self.errors = errors
        self.attempts = attempts


class StructuredOutcome(BaseModel):
    """What the caller needs for metrics: the object, and how hard it was to get."""

    attempts: int
    first_try: bool
    responses: list[LLMResponse]

    model_config = {"arbitrary_types_allowed": True}


def render_errors(exc: ValidationError) -> str:
    """Readable, located errors. ``applicant.national_id: must be 10 digits ...``"""
    lines = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"- {location}: {err['msg']}")
    return "\n".join(lines)


def extract_structured(  # noqa: UP047 - see the note on T above
    client: LLMClient,
    schema_model: type[T],
    *,
    system: str,
    user: str,
    schema_name: str,
    model_alias: str = "retail_support-extract",
    temperature: float = 0.0,
    max_tokens: int = 600,
    max_attempts: int = 2,
    response_format: dict | None = None,
) -> tuple[T, StructuredOutcome]:
    from retail_support.domain.order import strict_schema  # local import: avoids a cycle

    schema = response_format or strict_schema(schema_model, schema_name)
    messages = [Message(role="system", content=system), Message(role="user", content=user)]
    responses: list[LLMResponse] = []
    last_raw: str | None = None
    last_errors: list[dict] = []

    for attempt in range(1, max_attempts + 1):
        response = client.complete(
            LLMRequest(
                messages=messages,
                model_alias=model_alias,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=schema,
                cache_prefix_messages=1,
            )
        )
        responses.append(response)
        last_raw = response.text
        try:
            value = schema_model.model_validate_json(response.text or "")
        except ValidationError as exc:
            last_errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
            ]
            log.warning(
                "structured_validation_failed",
                schema=schema_name,
                attempt=attempt,
                errors=[e["loc"] for e in last_errors],
            )
            if attempt == max_attempts:
                break
            messages = messages + [
                Message(role="assistant", content=response.text or ""),
                Message(
                    role="user",
                    content=REPAIR_INSTRUCTION.format(errors=render_errors(exc)),
                ),
            ]
            continue
        except json.JSONDecodeError:  # pragma: no cover - JSON mode makes this rare
            last_errors = [{"loc": [], "msg": "output was not valid JSON", "type": "json"}]
            break
        else:
            log.info(
                "structured_extracted",
                schema=schema_name,
                attempt=attempt,
                outcome="first_try" if attempt == 1 else "after_repair",
            )
            return value, StructuredOutcome(
                attempts=attempt, first_try=attempt == 1, responses=responses
            )

    raise StructuredExtractionFailed(last_raw, last_errors, len(responses))
