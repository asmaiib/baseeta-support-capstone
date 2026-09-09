"""The FAQ handler: one call, a small model, a cacheable prompt.

The message assembly below *is* the Module 6 optimisation. Read the order:

    CACHEABLE PREFIX   the versioned system prompt + the service catalog,
                       byte-identical across every request in a language
    VOLATILE TAIL      windowed history, then this turn

Nothing dynamic above the fold. No timestamp, no session id, no "helpful context"
assembled per request. One dynamic byte early in the prompt invalidates everything
after it, and the classic self-inflicted wound is a courtesy: "Today is ..." at
the top of the system prompt. This handler keeps it in the volatile tail instead
(see the bottom of ``build_faq_messages``) — then look at ``cached_input_tokens``.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime

from retail_support.caching.response_cache import CacheScope, ResponseCache
from retail_support.domain.catalog import rendered_catalog
from retail_support.domain.session import Session
from retail_support.guards.input_guards import GuardedInput
from retail_support.llm.interfaces import LLMClient, LLMRequest, Message
from retail_support.observability import current_trace_id, get_logger
from retail_support.pipeline.groundedness import unsupported_amounts
from retail_support.pipeline.types import Reply
from retail_support.prompts.registry import PromptArtifact, load_prompt

log = get_logger(__name__)

#: Overridable so a walkthrough can serve a different prompt version without a
#: code edit, e.g. RETAIL_SUPPORT_FAQ_PROMPT=answer_faq.v2 once a v2 exists.
DEFAULT_FAQ_PROMPT = os.environ.get("RETAIL_SUPPORT_FAQ_PROMPT", "answer_faq.v1")


def build_faq_messages(
    prompt: PromptArtifact,
    catalog: str,
    history: list[Message],
    user_text: str,
    *,
    today: str | None = None,
    now: str | None = None,
) -> list[Message]:
    variables = {"product_catalog": catalog}
    if "today" in prompt.required_vars:
        variables["today"] = today or date.today().isoformat()
    if "now" in prompt.required_vars:
        # answer_faq.v4 asks for this, at the top of the prompt. It is a courtesy
        # that costs the entire cache discount on every request: one byte of the
        # prefix changes every second, so nothing after it can ever be reused.
        variables["now"] = now or datetime.now().isoformat(timespec="seconds")
    system = prompt.render(**variables)
    turn = f"<customer_message>\n{user_text}\n</customer_message>"
    if not {"today", "now"} & set(prompt.required_vars):
        # The date belongs in the volatile zone, with the turn — if it belongs
        # anywhere. Here it costs nothing; at the top of the prompt it costs the
        # entire cache discount on every request.
        turn = f"Today is {today or date.today().isoformat()}.\n{turn}"
    return [Message(role="system", content=system), *history, Message(role="user", content=turn)]


class FAQHandler:
    def __init__(
        self,
        client: LLMClient,
        *,
        model_alias: str = "retail_support-default",
        prompt_ref: str = DEFAULT_FAQ_PROMPT,
        cache: ResponseCache | None = None,
        meter=None,
        cascade_enabled: bool = False,
        cascade_alias: str = "retail_support-flagship",
    ) -> None:
        self._client = client
        self._model_alias = model_alias
        self._prompt = load_prompt(prompt_ref)
        self._cache = cache
        self._meter = meter
        self._cascade_enabled = cascade_enabled
        self._cascade_alias = cascade_alias
        self.escalations = 0

    @property
    def prompt_version(self) -> str:
        return self._prompt.ref

    def answer(self, guarded: GuardedInput, session: Session) -> Reply:
        t0 = time.perf_counter()
        language = guarded.language
        catalog = rendered_catalog("ar" if language == "ar" else "en")
        messages = build_faq_messages(
            self._prompt, catalog, session.state.messages(), guarded.text
        )
        params = {"temperature": 0.4, "max_tokens": 700, "alias": self._model_alias}

        scope = CacheScope(
            language=language,
            intent="faq",
            #: A question is impersonal only when nothing about this customer is in
            #: it. History makes a turn personal; so does anything the PII vault
            #: touched. Both are structural, not a judgement call at call time.
            personalised=bool(session.state.turns) or len(session.pii_vault) > 0,
        )
        key = ResponseCache.exact_key(
            self._model_alias, self._prompt.ref, guarded.text + catalog[:64], params
        )
        if self._cache is not None:
            hit = self._cache.get(scope=scope, query=guarded.text, exact_key=key)
            if hit is not None:
                payload, tier = hit
                return Reply(
                    text=payload["text"],
                    intent="faq",
                    language=language,
                    route=payload.get("route", ""),
                    model_id=payload.get("model_id", ""),
                    prompt_version=self._prompt.ref,
                    cache_tier=tier,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    trace_id=current_trace_id(),
                )

        response = self._client.complete(
            LLMRequest(
                messages=messages,
                model_alias=self._model_alias,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                cache_prefix_messages=1,
            )
        )
        text = response.text or ""
        if response.finish_reason == "length":
            # Truncation is a correctness bug, not a formatting quirk. Say so.
            log.warning("answer_truncated", model_id=response.model_id)

        cost_halalas = 0.0
        record = None
        if self._meter is not None:
            record = self._meter.meter(
                response,
                route=response.route or self._model_alias,
                intent="faq",
                stage="faq_handler",
                prompt_version=self._prompt.ref,
            )
            cost_halalas = record.cost_halalas

        # --- the cascade ------------------------------------------------
        # Cheap model first; escalate on a signal that is deterministic and
        # actually correlated with being wrong. "You quoted an amount that is not
        # in the catalog" is such a signal. "Was that hard?" is not — model
        # self-report is weak and sycophantic, and Module 6 says so out loud.
        #
        # Note the honest cost: an escalated request pays for both calls. The
        # cascade wins when escalations are rare, which is a property of your
        # traffic, which is why the meter comes before the optimisation.
        if self._cascade_enabled and unsupported_amounts(text, catalog):
            self.escalations += 1
            log.warning(
                "cascade_escalated",
                from_alias=self._model_alias,
                to_alias=self._cascade_alias,
                reason="unsupported_amount",
            )
            escalated = self._client.complete(
                LLMRequest(
                    messages=messages,
                    model_alias=self._cascade_alias,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    cache_prefix_messages=1,
                )
            )
            text = escalated.text or text
            response = escalated
            if self._meter is not None:
                record = self._meter.meter(
                    escalated,
                    route=escalated.route or self._cascade_alias,
                    intent="faq",
                    stage="faq_cascade",
                    prompt_version=self._prompt.ref,
                )
                cost_halalas += record.cost_halalas
        if self._cache is not None:
            self._cache.put(
                scope=scope,
                query=guarded.text,
                exact_key=key,
                value={"text": text, "model_id": response.model_id, "route": response.route},
            )
        return Reply(
            text=text,
            intent="faq",
            language=language,
            route=response.route,
            model_id=response.model_id,
            prompt_version=self._prompt.ref,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_halalas=round(cost_halalas, 6),
            input_tokens=response.usage.input_tokens,
            cached_tokens=response.usage.cached_input_tokens,
            output_tokens=response.usage.output_tokens,
            trace_id=current_trace_id(),
        )
