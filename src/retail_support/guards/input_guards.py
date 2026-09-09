"""Input guards, layered cheapest first.

    layer 1  deterministic   ~0.1 ms   length, normalisation, known payloads
    layer 2  PII masking     ~1 ms     mask before any model or log sees the text
    layer 3  classifier      ~300 ms   a small model, structured verdict

The honest position, stated in class and encoded here: **there is no
parameterised-query equivalent for prompt injection.** Instructions and data share
one token stream. This is layered mitigation, and it is measured in two numbers —
attack block rate *and* legitimate false-positive rate — because either one alone
can be gamed into a broken product.

Order matters twice over. Normalisation happens *before* pattern matching, or the
Arabic and homoglyph variants sail through. And the whole wall stands *before* the
router, so off-scope traffic never reaches the expensive route.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import unicodedata
from typing import Literal

from pydantic import BaseModel

from retail_support.domain.session import Session, mask_pii
from retail_support.guards.refusals import refusal_for
from retail_support.llm.interfaces import LLMClient
from retail_support.observability import get_logger
from retail_support.prompts.registry import load_prompt

log = get_logger(__name__)

#: Deliberately non-exhaustive, and bilingual for the same reason the corpus is:
#: an English-only blocklist on a bilingual product misses half the attack surface.
INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\bignore\s+(all\s+)?(your\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)",
        re.I,
    ),
    re.compile(r"\bdisregard\s+(all\s+)?(your\s+|the\s+)?(previous|prior|above)\b", re.I),
    re.compile(
        r"\b(reveal|show|print|repeat|output|display|tell me)\b.{0,40}"
        r"\b(system prompt|your instructions|developer message|initial prompt|your rules)\b",
        re.I,
    ),
    re.compile(
        r"\byou are (now|no longer)\b.{0,40}\b(assistant|admin|developer|unrestricted|dan)\b", re.I
    ),
    re.compile(r"\b(new|updated|revised) (system )?instructions?\s*:", re.I),
    re.compile(r"</?(system|instructions?)>", re.I),
    re.compile(r"\bact as (if you are |a )?(the )?(system|developer|administrator)\b", re.I),
    re.compile(
        r"\b(repeat|print|output|echo)\b.{0,30}\b(everything|all|the text|what is)\b.{0,20}\babove\b",
        re.I,
    ),
    re.compile(r"\btranslate your (system )?(prompt|instructions)\b", re.I),
    re.compile(r"تجاهل\s*(كل\s*)?(ال)?(تعليمات|أوامر|الأوامر|التعليمات)\s*(السابقة|السابق)?"),
    re.compile(
        r"(اطبع|أظهر|إظهار|اعرض|عرض|كشف|أعد|كرر|أخبرني)\s*.{0,25}"
        r"(تعليمات النظام|التعليمات السابقة|البرومبت|تعليماتك|قواعدك|إعداداتك)"
    ),
    re.compile(r"كرر\s*(كل\s*)?ما\s*(هو\s*)?(مكتوب|ورد)\s*(فوق|أعلى)"),
    re.compile(r"أنت\s*الآن\s*(مساعد|نظام|مطور|مدير)"),
    re.compile(r"تعليمات\s*(جديدة|محدثة)\s*:"),
    re.compile(r"تصرف\s*ك(نظام|مطور|مدير)"),
    # The second Module 4 miss: an authority claim carrying an instruction to drop
    # your own constraints. "Who is asking" is never the question — what is being
    # asked is. A ministry cannot ask an assistant to ignore its limits either.
    re.compile(r"تجاهل\s*(كل\s*)?(قيودك|حدودك|قواعدك|ضوابطك)"),
    re.compile(r"\bignore\s+(all\s+)?(your\s+)?(limits|constraints|restrictions|rules)\b", re.I),
]

#: The carve-out Module 4 earns: a customer asking for the *instructions
#: for a service* is a customer, not an attack. Without this line the guard blocks
#: three questions in the legitimate corpus and the false-positive rate is 5%.
LEGITIMATE_CONTEXT = re.compile(
    r"\b(instructions|steps|requirements|guide)\s+(for|to)\s+"
    r"(renew|renewing|issue|issuing|book|booking|apply|applying|update|updating|transfer)",
    re.I,
)

#: C0/C1 controls (tab and newline excepted), zero-width characters, and the
#: bidirectional overrides — the ones used to hide a payload inside text that
#: looks entirely innocent when rendered.
#: Built from ranges rather than written literally: a source file that contains
#: actual control characters is a source file no reviewer can review.
_CONTROL_RANGES = [
    (0x00, 0x08),  # C0 controls, keeping tab (09) and newline (0A)
    (0x0B, 0x0C),
    (0x0E, 0x1F),
    (0x7F, 0x9F),  # DEL and the C1 block
    (0x200B, 0x200F),  # zero-width space/joiners, LRM/RLM
    (0x202A, 0x202E),  # bidirectional embedding and override
    (0x2066, 0x2069),  # bidirectional isolates
    (0xFEFF, 0xFEFF),  # byte-order mark used mid-string
]
CONTROL_CHARS = re.compile(
    "[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _CONTROL_RANGES) + "]"
)


class GuardVerdict(BaseModel):
    allowed: bool = True
    layer: Literal["deterministic", "pii", "classifier", "none"] = "none"
    category: str = "ok"
    latency_ms: float = 0.0
    #: A hash, never the payload. Logs are not a place to keep attacks verbatim.
    payload_sha256: str = ""


class ScopeVerdict(BaseModel):
    """Guards are extraction — Module 3's machinery, reused wholesale."""

    category: Literal["ok", "injection_attempt", "off_scope", "crisis"]


class GuardedInput(BaseModel):
    original: str
    text: str  # normalised and PII-masked; this is what any model sees
    language: str
    verdict: GuardVerdict
    refusal: str | None = None

    @property
    def blocked(self) -> bool:
        return not self.verdict.allowed


ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")
LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    ar = len(ARABIC.findall(text))
    en = len(LATIN.findall(text))
    if ar and en:
        return "ar" if ar / (ar + en) >= 0.5 else "en"
    return "ar" if ar else "en"


def normalise(text: str) -> str:
    """NFKC folds homoglyph and presentation-form tricks into canonical characters.

    Running this *after* the pattern check instead of before it is the single most
    common way an Arabic attack row sails through a guard that looks correct.
    """
    return CONTROL_CHARS.sub("", unicodedata.normalize("NFKC", text))


def match_variants(text: str) -> list[str]:
    """The two readings of one string, because deleting is not the only option.

    ``Ignore<ZWSP>all<ZWSP>previous<ZWSP>instructions`` defeats a guard that strips
    zero-width characters, because stripping them welds the words together into
    ``Ignoreallpreviousinstructions`` and no word-boundary pattern matches. It
    equally defeats a guard that does nothing, because the raw string has no
    spaces either. The attack lives in the gap between two reasonable choices, so
    the guard stops choosing: patterns run against the deleted form *and* the form
    where each invisible separator became a space.

    This is one of the two misses recorded in Module 4 and fixed in Module 5. Note what
    the fix is not: it is not a new payload string added to a blocklist. Blocklists
    grow one attack at a time; this closes the shape.
    """
    folded = unicodedata.normalize("NFKC", text)
    return [CONTROL_CHARS.sub("", folded), CONTROL_CHARS.sub(" ", folded)]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def deterministic_checks(text: str, *, max_chars: int = 4000) -> GuardVerdict | None:
    """Layer 1. Returns a blocking verdict, or None to fall through to layer 2.

    Takes the **raw** input, not a pre-normalised copy: normalising once and then
    matching against that single reading is precisely the gap the zero-width
    payload lives in. Normalisation happens here, both ways, every time.
    """
    if len(text) > max_chars:
        return GuardVerdict(
            allowed=False,
            layer="deterministic",
            category="too_long",
            payload_sha256=_hash(text),
        )
    variants = match_variants(text)
    if any(LEGITIMATE_CONTEXT.search(candidate) for candidate in variants):
        return None
    for candidate in variants:
        for pattern in INJECTION_PATTERNS:
            if pattern.search(candidate):
                return GuardVerdict(
                    allowed=False,
                    layer="deterministic",
                    category="injection_pattern",
                    payload_sha256=_hash(text),
                )
    return None


DEFAULT_GUARD_PROMPT = os.environ.get("RETAIL_SUPPORT_GUARD_PROMPT", "input_guard_classifier.v1")


class InputGuard:
    """The wall. It stands before the router: one wall, all routes."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        max_chars: int = 4000,
        classifier_enabled: bool = True,
        classifier_alias: str = "retail_support-guard",
        prompt_ref: str = DEFAULT_GUARD_PROMPT,
        meter=None,
    ) -> None:
        self._client = client
        self._meter = meter
        self._max_chars = max_chars
        self._classifier_enabled = classifier_enabled and client is not None
        self._classifier_alias = classifier_alias
        self._prompt = load_prompt(prompt_ref)
        self.last_layer_timings: dict[str, float] = {}

    @property
    def prompt_version(self) -> str:
        return self._prompt.ref

    def check(self, text: str, session: Session) -> GuardedInput:
        t0 = time.perf_counter()
        language = detect_language(text)
        normalised = normalise(text)

        # The raw text, deliberately: the check normalises both ways itself.
        verdict = deterministic_checks(text, max_chars=self._max_chars)
        self.last_layer_timings = {"deterministic": (time.perf_counter() - t0) * 1000}
        if verdict is not None:
            verdict.latency_ms = self.last_layer_timings["deterministic"]
            log.warning(
                "guard_blocked",
                layer=verdict.layer,
                category=verdict.category,
                payload_sha256=verdict.payload_sha256,
            )
            return GuardedInput(
                original=text,
                text=normalised,
                language=language,
                verdict=verdict,
                refusal=refusal_for(verdict.category, language),
            )

        t1 = time.perf_counter()
        masked = mask_pii(normalised, session)
        self.last_layer_timings["pii"] = (time.perf_counter() - t1) * 1000

        if self._classifier_enabled:
            t2 = time.perf_counter()
            category = self._classify(masked)
            self.last_layer_timings["classifier"] = (time.perf_counter() - t2) * 1000
            if category != "ok":
                blocked = GuardVerdict(
                    allowed=False,
                    layer="classifier",
                    category=category,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    payload_sha256=_hash(normalised),
                )
                log.warning(
                    "guard_blocked",
                    layer="classifier",
                    category=category,
                    payload_sha256=blocked.payload_sha256,
                )
                return GuardedInput(
                    original=text,
                    text=masked,
                    language=language,
                    verdict=blocked,
                    refusal=refusal_for(category, language),
                )

        return GuardedInput(
            original=text,
            text=masked,
            language=language,
            verdict=GuardVerdict(allowed=True, latency_ms=(time.perf_counter() - t0) * 1000),
        )

    def _classify(self, text: str) -> str:
        from retail_support.pipeline.structured import extract_structured

        try:
            verdict, outcome = extract_structured(
                self._client,
                ScopeVerdict,
                system=self._prompt.render(),
                user=f"<customer_message>\n{text}\n</customer_message>",
                schema_name="guard_verdict",
                model_alias=self._classifier_alias,
                temperature=0.0,
                max_tokens=20,
            )
        except Exception as exc:  # noqa: BLE001 - a guard outage must not be silent
            # Failing closed here would take the product down on a provider blip;
            # failing open would remove the wall. The course's compromise: keep
            # layer 1's verdict (already passed), and log loudly enough to alert on.
            log.error("guard_classifier_unavailable", error=type(exc).__name__)
            return "ok"
        if self._meter is not None:
            # Meter coverage is 100% of model calls or it is not coverage. Guards
            # and routing are model calls, and on a routed pipeline they are a
            # visible share of the bill.
            for response in outcome.responses:
                self._meter.meter(
                    response,
                    route=response.route or self._classifier_alias,
                    intent="guard",
                    stage="input_guard",
                    prompt_version=self._prompt.ref,
                )
        return verdict.category
