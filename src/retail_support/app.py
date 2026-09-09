"""The composition root — the one place where concrete classes meet.

Everything else in ``src/retail_support`` depends on protocols and configuration. This
module is where a route name becomes an adapter, where the retry and fallback
policy gets wrapped around it, and where the five pipeline stages are assembled.
It is also the only module that would change if this application moved from the
course gateway to a real provider, an on-premise vLLM deployment, or all three at
once — and even then it changes by reading a different YAML file.
"""

from __future__ import annotations

import os
from pathlib import Path

from retail_support.caching.response_cache import ResponseCache, build_cache
from retail_support.config import Settings, get_settings
from retail_support.guards.input_guards import InputGuard
from retail_support.guards.output_guards import OutputGuard
from retail_support.llm.anthropic_client import AnthropicClient
from retail_support.llm.interfaces import LLMClient
from retail_support.llm.openai_compat import OpenAICompatClient
from retail_support.llm.resilient import ResilientClient
from retail_support.observability import configure_logging, get_logger
from retail_support.observability.cost import CostMeter
from retail_support.pipeline.assemble import Dependencies, EscalationHandler, RetailSupport
from retail_support.pipeline.faq import DEFAULT_FAQ_PROMPT, FAQHandler
from retail_support.pipeline.router import IntentRouter
from retail_support.pipeline.service import ServiceWorkflow

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_COST_LOG = PROJECT_ROOT / "logs" / "llm_cost.jsonl"


def build_client(settings: Settings, route_name: str) -> LLMClient:
    """A route name becomes an adapter. This function is the provider abstraction
    doing its job: three deployments, two dialects, one return type."""
    route = settings.route(route_name)
    if route.dialect == "anthropic":
        return AnthropicClient(route)
    if route.dialect == "fake":
        from retail_support.llm.fake import FakeClient
        from retail_support.llm.fake_brain import smart_default

        return FakeClient(route=route_name).always(smart_default)
    return OpenAICompatClient(route)


def build_clients(settings: Settings | None = None) -> dict[str, LLMClient]:
    settings = settings or get_settings()
    return {name: build_client(settings, name) for name in settings.routes}


def build_resilient(
    settings: Settings, *, primary: str | None = None, fallback: str | None = None
) -> ResilientClient:
    primary = primary or settings.primary_route
    hops: list[tuple[str, LLMClient]] = [("primary", build_client(settings, primary))]
    fallback = fallback if fallback is not None else settings.fallback_route
    if fallback and fallback != primary and fallback in settings.routes:
        hops.append((f"fallback:{fallback}", build_client(settings, fallback)))
    return ResilientClient(hops)


def build_assistant(
    settings: Settings | None = None,
    *,
    route: str | None = None,
    meter: CostMeter | None = None,
    cache: ResponseCache | None = None,
    cost_log: str | Path | None = None,
) -> RetailSupport:
    """Wire the whole application.

    ``route`` pins every model call to one configured route — that is how
    ``make ask ROUTE=vllm`` sends the same code to the open-weight model with no
    edit anywhere, and how the provider comparison in Module 2 is a loop rather
    than a branch.
    """
    settings = settings or get_settings()
    configure_logging()

    if route:
        client: LLMClient = build_client(settings, route)
        cheap: LLMClient = client
    else:
        client = build_resilient(settings)
        cheap = build_client(settings, settings.cheap_route)

    meter = meter or CostMeter(settings.prices, sink=cost_log or os.environ.get("RETAIL_SUPPORT_COST_LOG"))
    cache = cache if cache is not None else (build_cache(settings) if settings.cache.enabled else None)

    table = settings.pipeline.routing_table if settings.pipeline.routing_enabled else {}
    faq_alias = table.get("faq") or "retail_support-default"
    service_alias = table.get("service") or "retail_support-default"

    deps = Dependencies(
        input_guard=InputGuard(
            cheap,
            max_chars=settings.guards.max_input_chars,
            classifier_enabled=settings.guards.classifier_enabled,
            classifier_alias=settings.guards.classifier_alias,
            meter=meter,
        ),
        router=IntentRouter(cheap, meter=meter),
        faq_handler=FAQHandler(
            client,
            model_alias=faq_alias,
            prompt_ref=DEFAULT_FAQ_PROMPT,
            cache=cache,
            meter=meter,
            cascade_enabled=settings.pipeline.cascade_enabled,
            cascade_alias=settings.pipeline.cascade_escalate_alias,
        ),
        service_workflow=ServiceWorkflow(
            client,
            model_alias=service_alias,
            max_iterations=settings.pipeline.max_tool_iterations,
            meter=meter,
        ),
        output_guard=OutputGuard(settings.guards.canary),
        escalation=EscalationHandler(),
        routing_enabled=settings.pipeline.routing_enabled,
        routing_table=settings.pipeline.routing_table,
    )
    assistant = RetailSupport(deps)
    assistant.meter = meter  # type: ignore[attr-defined]
    assistant.cache = cache  # type: ignore[attr-defined]
    assistant.settings = settings  # type: ignore[attr-defined]
    log.info(
        "assistant_built",
        route=route or f"{settings.primary_route}+fallback",
        faq_alias=faq_alias,
        service_alias=service_alias,
        routing_enabled=settings.pipeline.routing_enabled,
        cascade=settings.pipeline.cascade_enabled,
        cache=bool(cache),
        semantic_cache=bool(cache and cache.semantic_enabled),
        faq_prompt=DEFAULT_FAQ_PROMPT,
    )
    return assistant
