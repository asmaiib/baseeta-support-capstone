"""The whole RetailSupport request path as five named, individually testable stages.

    guard_input | route_intent | (faq | service | escalate) | guard_output

LangChain supplies composition, batching and tracing hooks. The *logic* stays in
plain Python objects that can be constructed and called in a unit test with no
framework involved — which is the point. Frameworks churn; boundaries endure. The
``LLMClient`` boundary sits underneath all of this rather than being replaced by
the framework's own abstraction, and that is a deliberate, defensible choice.

The rule that keeps this honest: **if a stage cannot be run alone in a test, it is
not a stage.** ``tests/pipeline/test_stages.py`` runs each of the five with stubs
and no network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda

from retail_support.domain.session import Session
from retail_support.guards.input_guards import GuardedInput, InputGuard
from retail_support.guards.output_guards import OutputGuard
from retail_support.guards.refusals import refusal_for
from retail_support.observability import get_logger, new_trace_id
from retail_support.pipeline.faq import FAQHandler
from retail_support.pipeline.router import IntentRouter
from retail_support.pipeline.service import ServiceWorkflow
from retail_support.pipeline.types import Reply

log = get_logger(__name__)


@dataclass
class Request:
    """What flows into the pipeline."""

    text: str
    session: Session


@dataclass
class Routed:
    """What flows between the router and the handlers."""

    guarded: GuardedInput
    session: Session
    intent: str = "faq"


class EscalationHandler:
    """Humans are not a model call — so this stage makes none."""

    def __init__(self, escalation_service=None) -> None:
        if escalation_service is None:
            from retail_support.tools.services import escalation_service as default

            escalation_service = default
        self._service = escalation_service

    def handoff(self, routed: Routed) -> Reply:
        record = self._service.handoff(
            reason="router sent this conversation to a human", session=routed.session
        )
        language = routed.guarded.language
        text = (
            "سأحوّلك إلى موظف خدمة الآن، وسيصلك رد في أقرب وقت."
            if language == "ar"
            else "I'm transferring you to a human agent now; they'll be with you shortly."
        )
        return Reply(
            text=text,
            intent="escalate",
            language=language,
            escalated=True,
            tool_calls=[{"tool": "escalate_to_agent", "risk": "terminal", "args": record}],
        )


@dataclass
class Dependencies:
    input_guard: InputGuard
    router: IntentRouter
    faq_handler: FAQHandler
    service_workflow: ServiceWorkflow
    output_guard: OutputGuard
    escalation: EscalationHandler
    routing_enabled: bool = False
    routing_table: dict[str, str | None] | None = None


def build_pipeline(deps: Dependencies) -> Runnable:
    def guard_input(request: Request) -> Routed | Reply:
        new_trace_id()
        guarded = deps.input_guard.check(request.text, request.session)
        if guarded.blocked:
            return Reply(
                text=guarded.refusal or refusal_for("off_scope", guarded.language),
                intent="refused",
                language=guarded.language,
                blocked=True,
                guard_layer=guarded.verdict.layer,
                guard_category=guarded.verdict.category,
                latency_ms=guarded.verdict.latency_ms,
            )
        return Routed(guarded=guarded, session=request.session)

    def route_intent(value: Routed | Reply) -> Routed | Reply:
        if isinstance(value, Reply):
            return value
        value.intent = deps.router.classify(value.guarded.text)
        return value

    def run_faq(value: Routed) -> Reply:
        return deps.faq_handler.answer(value.guarded, value.session)

    def run_service(value: Routed) -> Reply:
        return deps.service_workflow.run(value.guarded, value.session)

    def guard_output(reply: Reply) -> Reply:
        if reply.blocked:
            return reply
        text, verdict = deps.output_guard.apply(reply.text)
        reply.output_guard_category = verdict.category
        if not verdict.allowed:
            log.warning("output_guard_blocked", category=verdict.category)
            reply.text = text
            reply.blocked = True
        return reply

    handler = RunnableBranch(
        (lambda v: isinstance(v, Reply), RunnableLambda(lambda v: v)),
        (lambda v: v.intent == "faq", RunnableLambda(run_faq)),
        (lambda v: v.intent == "service", RunnableLambda(run_service)),
        RunnableLambda(deps.escalation.handoff),
    )

    return (
        RunnableLambda(guard_input)
        | RunnableLambda(route_intent)
        | handler
        | RunnableLambda(guard_output)
    )


class RetailSupport:
    """A thin façade over the pipeline: what the CLI, the API and the harness use."""

    def __init__(self, deps: Dependencies) -> None:
        self.deps = deps
        self.pipeline = build_pipeline(deps)

    def ask(self, text: str, session: Session, *, remember: bool = True) -> Reply:
        t0 = time.perf_counter()
        reply: Reply = self.pipeline.invoke(Request(text=text, session=session))
        reply.latency_ms = (time.perf_counter() - t0) * 1000
        if remember and not reply.blocked:
            session.state.add_user(text)
            session.state.add_assistant(reply.text)
        return reply
