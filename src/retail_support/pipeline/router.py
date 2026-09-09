"""The router: the workhorse pattern, and the biggest cost lever in an assistant.

RetailSupport's traffic is roughly 70% FAQ, 25% transactional, 5% escalation. Routing is
what makes the economics work — and because a routing change moves quality, it is
**eval-gated exactly like a prompt change**. Module 6 turns the table below on;
the gate decides whether it stays on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from retail_support.llm.interfaces import LLMClient
from retail_support.observability import get_logger
from retail_support.pipeline.structured import extract_structured
from retail_support.prompts.registry import load_prompt

log = get_logger(__name__)

Intent = Literal["faq", "service", "escalate"]

#: Config, not code — reviewed alongside the price sheet. ``None`` means "not a
#: model call at all": humans are not a route.
ROUTING_TABLE: dict[str, str | None] = {
    "faq": "retail_support-small",
    "service": "retail_support-default",
    "complex": "retail_support-flagship",
    "escalate": None,
}


class RouteVerdict(BaseModel):
    intent: Intent


class IntentRouter:
    def __init__(
        self,
        classifier: LLMClient,
        *,
        model_alias: str = "retail_support-router",
        prompt_ref: str = "route_intent.v1",
        meter=None,
    ) -> None:
        self._classifier = classifier
        self._meter = meter
        self._model_alias = model_alias
        self._prompt = load_prompt(prompt_ref)

    @property
    def prompt_version(self) -> str:
        return self._prompt.ref

    def classify(self, text: str) -> Intent:
        try:
            verdict, outcome = extract_structured(
                self._classifier,
                RouteVerdict,
                system=self._prompt.render(),
                user=f"<customer_message>\n{text}\n</customer_message>",
                schema_name="route_verdict",
                model_alias=self._model_alias,
                temperature=0.0,
                max_tokens=20,
            )
        except Exception as exc:  # noqa: BLE001
            # A router that cannot decide sends traffic to the handler that can
            # cope with anything. Failing to the expensive route costs money;
            # failing to the cheap one costs correctness. Choose deliberately.
            log.warning("router_unavailable", error=type(exc).__name__, fallback="service")
            return "service"
        if self._meter is not None:
            for response in outcome.responses:
                self._meter.meter(
                    response,
                    route=response.route or self._model_alias,
                    intent="router",
                    stage="router",
                    prompt_version=self._prompt.ref,
                )
        log.info("routed", intent=verdict.intent, prompt_version=self._prompt.ref)
        return verdict.intent
