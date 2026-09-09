"""The course's one agentic loop — bounded, authorised, traced.

Five sentences hold the whole security model:

1. The model **requests**; the application **executes**. Nothing else.
2. Read-only tools run freely. Side-effecting tools pass the authorisation gate.
3. The gate checks the *session*, never the model's arguments — a ``national_id``
   in a tool call is user input by proxy, and it may have been injected.
4. Domain errors go back to the model as data so it can recover conversationally.
   Bugs do not: a stack trace leaks implementation detail and confuses the model.
5. The loop is bounded. A failing tool retried thirty times is a denial-of-wallet
   attack you wrote yourself.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import ValidationError

from retail_support.domain.session import Session
from retail_support.llm.interfaces import LLMClient, LLMRequest, LLMResponse, Message
from retail_support.observability import get_logger
from retail_support.tools.registry import BY_NAME, Tool, tool_schemas
from retail_support.tools.services import ToolError

log = get_logger(__name__)

MAX_ITERATIONS = 6

#: Argument contracts for side-effecting tools, checked before anything happens.
#: Syntax came from the schema; these are the semantics the schema cannot express.
ARGUMENT_MODELS: dict[str, Callable] = {}


def _argument_models() -> dict:
    if not ARGUMENT_MODELS:
        from retail_support.domain.order import ReturnRequest

        ARGUMENT_MODELS["create_return_request"] = ReturnRequest
    return ARGUMENT_MODELS


BOUND_HIT_MESSAGE = {
    "en": (
        "I could not complete this automatically — I am transferring you to an "
        "agent who can help."
    ),
    "ar": "لم أتمكن من إتمام طلبك آلياً — سأحوّلك إلى موظف يستطيع مساعدتك.",
}


class ToolLoopResult:
    def __init__(self) -> None:
        self.text: str = ""
        self.iterations: int = 0
        self.responses: list[LLMResponse] = []
        self.calls: list[dict] = []
        self.bound_hit: bool = False
        self.escalated: bool = False


def run_with_tools(
    client: LLMClient,
    messages: list[Message],
    session: Session,
    *,
    allowed_tools: list[str] | None = None,
    model_alias: str = "retail_support-default",
    max_iterations: int = MAX_ITERATIONS,
    max_tokens: int = 800,
    language: str = "en",
) -> ToolLoopResult:
    result = ToolLoopResult()
    schemas = tool_schemas(allowed_tools)
    working = list(messages)

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        response = client.complete(
            LLMRequest(
                messages=working,
                model_alias=model_alias,
                max_tokens=max_tokens,
                temperature=0.2,
                tools=schemas,
                cache_prefix_messages=1,
            )
        )
        result.responses.append(response)

        if response.finish_reason != "tool_calls" or not response.tool_calls:
            result.text = response.text or ""
            return result

        working.append(
            Message(
                role="assistant",
                content=response.text or "",
                tool_calls=response.tool_calls,
            )
        )
        # Read-only calls in one turn are independent and may run concurrently.
        # Side-effecting ones serialise behind the gate — always, even when the
        # model asked for them in the same breath.
        for call in response.tool_calls:
            tool = BY_NAME.get(call.name)
            payload = _execute(tool, call, session, iteration, result)
            working.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    name=call.name,
                    content=json.dumps(payload, ensure_ascii=False),
                )
            )
            if tool is not None and tool.risk == "terminal":
                result.escalated = True

    log.error("tool_loop_bound_hit", iterations=max_iterations)
    result.bound_hit = True
    result.text = BOUND_HIT_MESSAGE.get(language, BOUND_HIT_MESSAGE["en"])
    return result


def _execute(
    tool: Tool | None, call, session: Session, iteration: int, result: ToolLoopResult
) -> dict:
    if tool is None:
        log.warning("tool_unknown", requested=call.name, iteration=iteration)
        return {"error": "unknown_tool", "hint": "That tool does not exist. Use one of the listed tools."}

    try:
        args = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        log.warning("tool_malformed_arguments", tool=tool.name, iteration=iteration)
        return {"error": "malformed_arguments", "hint": "Send valid JSON matching the schema."}
    if not isinstance(args, dict):
        return {"error": "malformed_arguments", "hint": "Arguments must be a JSON object."}

    # Unmask inside the trust boundary: the tool needs the real value, the model
    # never saw it, and nothing masked leaves this function.
    args = {k: session.pii_vault.unmask(v) if isinstance(v, str) else v for k, v in args.items()}

    model = _argument_models().get(tool.name)
    if model is not None:
        try:
            validated = model(**args)
        except ValidationError as exc:
            reason = exc.errors()[0]["msg"] if exc.errors() else "invalid arguments"
            log.warning("tool_arguments_invalid", tool=tool.name, reason=reason)
            return {"error": "invalid_arguments", "hint": reason}
        except TypeError as exc:
            return {"error": "invalid_arguments", "hint": str(exc)}
        args = {
            k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in validated.model_dump().items()
        }

    if tool.risk == "side_effecting":
        replay = session.replay_side_effect(tool.name, args)
        if replay is not None:
            log.info("tool_idempotent_replay", tool=tool.name)
            return replay
        verdict = session.authorize(tool.name, args)
        if not verdict.allowed:
            log.warning("tool_denied", tool=tool.name, reason=verdict.reason)
            return {"error": "not_authorized", "reason": verdict.reason, "hint": verdict.user_hint}

    entry = {"tool": tool.name, "risk": tool.risk, "iteration": iteration, "args": args}
    session.tool_trace.append(entry)
    result.calls.append(entry)
    log.info("tool_call", tool=tool.name, iteration=iteration, risk=tool.risk)

    try:
        payload = tool.fn(**args, session=session)
    except ToolError as exc:  # domain errors are model input
        log.info("tool_domain_error", tool=tool.name, code=exc.code)
        return {"error": exc.code, "hint": exc.hint}
    except Exception:  # bugs are NOT model input
        log.exception("tool_crashed", tool=tool.name)
        return {"error": "temporarily_unavailable", "hint": "Try again in a moment."}

    if tool.risk == "side_effecting":
        session.record_side_effect(tool.name, args, payload)
    return payload
