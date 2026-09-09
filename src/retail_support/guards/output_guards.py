"""The outbound wall. The model's output is untrusted too.

Deterministic checks first, because they are free and because safety-critical
assertions must never depend on a judgement call:

* **canary** — a marker planted in every system prompt by the registry. If it
  appears in a response, the system prompt leaked. One string comparison, zero
  latency, and it catches the entire class.
* **outbound PII** — the same Saudi patterns as the inbound guard, applied the
  other way. Nothing unmasked leaves.
* **tool internals** — a stack trace or a driver error reaching a customer is a
  leak and a bad experience at the same time.
* **indirect injection relay** — content that arrived from a tool result or a
  document is not allowed to smuggle an instruction out through the answer. This
  is the wall that catches the poisoned ``note`` field, and the reason it exists
  one module before anyone says the word "retrieval".
"""

from __future__ import annotations

import re

from retail_support.domain.session import SAUDI_PII, Session
from retail_support.guards.input_guards import GuardVerdict, detect_language
from retail_support.guards.refusals import refusal_for
from retail_support.observability import get_logger
from retail_support.prompts.registry import CANARY

log = get_logger(__name__)

#: Implementation detail that must never reach a customer.
INTERNALS = [
    re.compile(r"\bTraceback \(most recent call last\)", re.I),
    re.compile(r"\b(psycopg2|sqlalchemy|httpx|urllib3)\.[A-Za-z]*Error\b"),
    re.compile(r"\b[A-Za-z]*(ConnectionError|OperationalError|TimeoutError)\b"),
    re.compile(r"\bstack trace\b", re.I),
]

#: Instructions that arrived from a tool result or a document, relayed outward.
RELAYED_INSTRUCTIONS = [
    re.compile(r"\bas the assistant reading this\b", re.I),
    re.compile(r"\b(call|dial|contact) (this|the following) number\b", re.I),
    re.compile(r"\bignore (the|your) (previous|prior) instructions\b", re.I),
    re.compile(r"اتصل\s*(على|ب)\s*(هذا\s*)?الرقم"),
]


class OutputGuard:
    def __init__(self, canary: str = CANARY) -> None:
        self._canary = canary

    def check(self, text: str, session: Session | None = None) -> GuardVerdict:
        language = detect_language(text)
        if self._canary in text:
            log.error("output_guard_leak", category="system_prompt_leak")
            return GuardVerdict(allowed=False, layer="deterministic", category="system_prompt_leak")
        for kind, pattern in SAUDI_PII.items():
            if pattern.search(text):
                log.error("output_guard_leak", category=f"pii_outbound_{kind}")
                return GuardVerdict(allowed=False, layer="pii", category="pii_outbound")
        for pattern in INTERNALS:
            if pattern.search(text):
                log.error("output_guard_leak", category="internal_error_leak")
                return GuardVerdict(
                    allowed=False, layer="deterministic", category="unavailable"
                )
        for pattern in RELAYED_INSTRUCTIONS:
            if pattern.search(text):
                log.error("output_guard_leak", category="relayed_instruction")
                return GuardVerdict(
                    allowed=False, layer="deterministic", category="unavailable"
                )
        _ = language
        return GuardVerdict(allowed=True)

    def apply(self, text: str, session: Session | None = None) -> tuple[str, GuardVerdict]:
        """Returns the text to send — the answer, or its designed refusal."""
        verdict = self.check(text, session)
        if verdict.allowed:
            return text, verdict
        return refusal_for(verdict.category, detect_language(text)), verdict
