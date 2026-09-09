"""A compact, deterministic "brain" for the zero-key ``fake`` route.

FakeClient (llm/fake.py) is a pure scripting harness: an empty queue raises
loudly, by design, so unit tests catch an unscripted call. That is correct
for tests, and insufficient for a CLI session, a notebook demo, or an eval
harness run across 100+ golden cases, none of which want to pre-script every
turn.

This module is that missing piece: a single ``smart_default(request)``
function, wired in via ``FakeClient(...).always(smart_default)`` in
``app.build_client``. It inspects the request's schema/tools to work out
which pipeline stage is asking (router, guard classifier, extractor, judge,
service workflow, or plain FAQ) and answers deterministically, grounded in
the real product catalog — the same discipline the course's own mockgw
simulator uses, at a fraction of the size, scoped to what this repository's
own golden set actually exercises. It is not a model; see BENCHMARKS.md and
the Evaluation Report's known-limitations section for what that means for
these numbers.
"""

from __future__ import annotations

import json
import re

from retail_support.domain.catalog import load_catalog
from retail_support.llm.interfaces import LLMRequest, LLMResponse, ToolCall, Usage

_ORDER_REF = re.compile(r"\bORD-[0-9]{6,10}\b", re.IGNORECASE)
_DATE = re.compile(r"\b(20[0-9]{2}-[0-9]{2}-[0-9]{2})\b")
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_PHONE = re.compile(r"(?:\+?9665|05|9665)\d{7,8}")
#: Arabic diacritics and tatweel — two spellings of one word (أكد/أكّد) must
#: match the same trigger, exactly the normalisation caching/embeddings.py
#: already applies to cache keys.
_DIACRITICS = re.compile("[" + "".join(chr(c) for c in range(0x064B, 0x0653)) + "ـ]")


def _norm(text: str) -> str:
    return _DIACRITICS.sub("", text)


_CATEGORY_KEYWORDS = {
    "electronics": ["laptop", "phone", "charger", "electronics", "لابتوب", "جوال", "شاحن", "الكترونيات", "إلكترونيات", "سماعات"],
    "home_goods": ["furniture", "kitchen", "blender", "home", "أثاث", "مطبخ", "أواني", "منزل"],
    "fashion": ["jacket", "shoes", "shoe", "clothing", "fashion", "جاكيت", "حذاء", "أحذية", "ملابس", "أزياء"],
}
_OUT_OF_DOMAIN_WORDS = [
    "grocery", "groceries", "car", "jewellery", "jewelry", "pet", "book", "gift card", "plant",
    "غذائية", "سيارة", "مجوهرات", "حيوانات", "كتب", "هدايا", "نباتات",
]
_CITY_NAMES = {
    "Riyadh": ["riyadh", "الرياض"], "Jeddah": ["jeddah", "جدة"], "Dammam": ["dammam", "الدمام"],
    "Makkah": ["makkah", "مكة"], "Madinah": ["madinah", "المدينة"], "Khobar": ["khobar", "الخبر"],
}
_ESCALATE_WORDS = ["human", "agent", "representative", "complaint", "escalat", "person", "put me through", "someone", "موظف", "شكوى", "مسؤول", "ممثل"]
_URGENT_WORDS = ["urgent", "asap", "emergency", "immediately", "عاجل", "بسرعة", "فوري"]
_RETURN_WORDS = ["return", "refund", "money back", "استرجاع", "استرداد", "إرجاع", "ارجاع"]
_EXCHANGE_WORDS = ["exchange", "swap", "replace", "replacement", "استبدال", "استبدل", "البديل", "بديل"]
_STATUS_WORDS = ["status", "where", "track", "check", "شحن", "تتبع", "حالة", "وين", "استعلم", "تستعلم", "استعلام"]
_CONFIRM_WORDS = ["yes", "confirm", "i confirm", "نعم", "أكد", "اكد"]
_OFF_SCOPE_WORDS = ["medicine", "headache", "vote", "election", "homework", "invest", "stock market", "buy stock", "دواء", "انتخاب", "استثمار", "أسهم"]
_CRISIS_WORDS = ["kill myself", "suicide", "hurt myself", "أنهي حياتي", "أأذي نفسي"]
_INJECTION_MARKERS = [
    "ignore all previous", "system prompt", "you are now", "your configuration",
    "print your instructions", "has approved this", "reveal your",
    "internal reference codes", "internal codes",
    "تجاهل", "تعليمات النظام", "أنت الآن",
]


def _last_user_text(request: LLMRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            inner = re.search(r"<customer_message>\s*(.*?)\s*</customer_message>", message.content, re.S)
            return (inner.group(1) if inner else message.content).strip()
    return ""


def _all_text(request: LLMRequest) -> str:
    return " ".join(m.content for m in request.messages if m.content)


def _user_text(request: LLMRequest) -> str:
    """Only what the customer actually said, across every turn — never the
    system prompt (which carries today's date and the catalog, both of which
    can contain dates/keywords that must not be mistaken for user input)."""
    return " ".join(m.content for m in request.messages if m.role == "user" and m.content)


def _language(text: str) -> str:
    # Order references (ORD-1234567) are identifiers, not English content —
    # strip them before deciding, or every Arabic message with an order
    # number reads as "mixed".
    stripped = _ORDER_REF.sub("", text)
    has_ar = bool(_ARABIC.search(stripped))
    has_en = bool(re.search(r"[A-Za-z]{3,}", stripped))
    if has_ar and has_en:
        return "mixed"
    return "ar" if has_ar else "en"


def _category(text: str) -> str:
    for cat, words in _CATEGORY_KEYWORDS.items():
        if _contains_any(text, words):
            return cat
    return "other"


def _city(text: str) -> str:
    for city, words in _CITY_NAMES.items():
        if _contains_any(text, words):
            return city
    return "unknown"


def _contains_any(text: str, words: list[str]) -> bool:
    low = _norm(text.lower())
    return any(_norm(w.lower()) in low for w in words)


def _catalog_field_text(field: str, category: str, language: str) -> str | None:
    catalog = load_catalog()
    if category in ("electronics", "home_goods", "fashion"):
        entry = catalog.by_category(category)
        if entry is not None and hasattr(entry, field):
            return getattr(entry, field).get(language)
    return None


def _faq_field(text: str) -> str | None:
    """Which catalog field this question is about, independent of category —
    checked before deciding whether the category itself is knowable, so a
    field-shaped question never falls through to "I don't know" just because
    it didn't happen to name a product.
    """
    if _contains_any(text, ["exchange", "swap", "replace", "replacement", "استبدال", "استبدل", "البديل", "بديل"]):
        return "exchange"  # general_policy, not a category field
    if _contains_any(text, ["restock", "open the box", "إعادة تخزين", "فتحت الصندوق"]):
        return "restocking_fee"
    if _contains_any(text, ["warrant", "ضمان"]):
        return "warranty"
    if _contains_any(text, ["ship", "deliver", "شحن", "توصيل"]):
        return "shipping_fee"
    if _contains_any(text, ["return", "days", "window", "إرجاع", "استرجاع", "ارجاع", "يوم", "مدة"]):
        return "return_window"
    return None


def _faq_answer(text: str, language: str) -> str:
    catalog = load_catalog()
    lang = "ar" if language == "ar" else "en"

    # General-policy questions first (not tied to a category).
    if _contains_any(text, ["refund", "money back", "store credit", "استرداد", "المبلغ", "رصيد"]):
        return catalog.general_policy["refund_method"].get(lang)
    field = _faq_field(text)
    if field == "exchange":
        return catalog.general_policy["exchange"].get(lang)
    if _contains_any(text, ["drop off", "drop-off", "courier", "pick up", "تسليم", "استلام", "شركة الشحن", "زيارة الفرع", "نفس الفرع"]):
        return catalog.general_policy["store_pickup"].get(lang)
    if _contains_any(text, ["which cities", "which city", "store locations", "أي مدن", "الفروع"]):
        return "Baseeta stores: " + ", ".join(catalog.store_locations.get(lang))

    category = _category(text)
    if category == "other":
        if field is None or _contains_any(text, _OUT_OF_DOMAIN_WORDS):
            return (
                "لا تتوفر لدي هذه المعلومة في كتالوج المنتجات، ولن أخمّن. يرجى مراجعة أقرب فرع."
                if lang == "ar"
                else "I don't have that information in the product catalog, so I won't guess. Please check with your nearest store."
            )
        # A field-shaped question (return window, warranty, shipping...) that
        # didn't name a specific product — answer for the flagship category
        # rather than refusing a perfectly answerable question.
        category = "electronics"

    entry = catalog.by_category(category)
    return getattr(entry, field or "return_window").get(lang)


def _route_verdict(text: str) -> dict:
    if _contains_any(text, _ESCALATE_WORDS):
        return {"intent": "escalate"}
    if _ORDER_REF.search(text):
        return {"intent": "service"}
    action_phrases = [
        "i want to", "i need to", "please file", "please book", "i'd like to",
        "can you", "could you", "would you", "please",
        "أبغى", "أريد", "احجز", "افتح لي", "سوّي لي", "يمكنك", "تقدر", "ممكن", "من فضلك",
    ]
    is_action = _contains_any(text, action_phrases) and _contains_any(
        text, _RETURN_WORDS + _EXCHANGE_WORDS + _STATUS_WORDS
    )
    if is_action:
        return {"intent": "service"}
    if "?" in text or "؟" in text:
        # Order-reference and imperative-action cases are already handled
        # above; anything left that ends in a question mark is, by
        # elimination, an informational policy question.
        return {"intent": "faq"}
    if _contains_any(text, _RETURN_WORDS + _EXCHANGE_WORDS + _STATUS_WORDS):
        return {"intent": "service"}
    return {"intent": "faq"}


def _guard_verdict(text: str, system_prompt: str = "") -> dict:
    # The v0 classifier (see prompts/library/input_guard_classifier/v0.md)
    # has no carve-out for "what are the instructions for X" — it blocks any
    # mention of "instructions" outright. That is a real, seeded regression:
    # it trips the legitimate-corpus traps this repository's golden set
    # carries specifically to catch it.
    has_carveout = "carve-out" in system_prompt.lower() or "carve out" in system_prompt.lower()
    if not has_carveout and _contains_any(text, ["instructions", "rules", "تعليمات", "قواعد"]):
        return {"category": "injection_attempt"}
    if _contains_any(text, _INJECTION_MARKERS):
        return {"category": "injection_attempt"}
    if _contains_any(text, _CRISIS_WORDS):
        return {"category": "crisis"}
    if _contains_any(text, _OFF_SCOPE_WORDS):
        return {"category": "off_scope"}
    return {"category": "ok"}


def _extract_return_case(text: str) -> dict:
    language = _language(text)
    order_ref = _ORDER_REF.search(text)
    phone = _PHONE.search(text.replace(" ", ""))
    is_ar = _ARABIC.search(text) is not None
    m = re.search(r"(?:my name is|i am|i'm)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", text)
    m_ar = re.search(r"(?:اسمي|أنا)\s+([\u0621-\u064A]+(?:\s+[\u0621-\u064A]+)?)", text)
    # full_name is a REQUIRED field in the schema (never null) — the extraction
    # prompt's never-invent rule applies to what goes IN the field, not to
    # whether the field itself is present, so an unnamed message still needs
    # a placeholder, exactly like the golden set's own "Unnamed customer".
    if m:
        name = m.group(1)
    elif m_ar:
        name = m_ar.group(1)
    else:
        name = "عميل" if is_ar else "Unnamed customer"
    issue_type = "other"
    if _contains_any(text, _EXCHANGE_WORDS):
        issue_type = "exchange"
    elif _contains_any(text, _RETURN_WORDS):
        issue_type = "return"
    elif _contains_any(text, _STATUS_WORDS) and order_ref:
        issue_type = "order_status"
    elif _contains_any(text, ["complaint", "unacceptable", "weeks", "شكوى", "أسابيع"]):
        issue_type = "complaint"
    urgency = "routine"
    if _contains_any(text, ["emergency", "طارئ"]):
        urgency = "emergency"
    elif _contains_any(text, _URGENT_WORDS):
        urgency = "urgent"
    return {
        "issue_type": issue_type,
        "order_reference": order_ref.group(0).upper() if order_ref else None,
        "category": _category(text),
        "summary_en": (text[:117] + "...") if len(text) > 120 else text,
        "city": _city(text),
        "urgency": urgency,
        "language": language,
        "customer": {"full_name": name, "phone": phone.group(0) if phone else None},
        "needs_human": _contains_any(text, _ESCALATE_WORDS),
    }


def _judge_verdict(text: str) -> dict:
    rub = re.search(r"<rubric>\s*(.*?)\s*</rubric>", text, re.S)
    ctx = re.search(r"<context>\s*(.*?)\s*</context>", text, re.S)
    ans = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
    rubric_text = rub.group(1) if rub else ""
    context_text = ctx.group(1) if ctx else ""
    answer_text = ans.group(1) if ans else ""
    from retail_support.pipeline.groundedness import is_grounded, unsupported_amounts

    if not answer_text.strip():
        return {"score": 0.0, "evidence": "empty answer"}

    # A rubric that requires a quoted evidence line and states the
    # don't-know clause explicitly (v2-shaped) gets careful, anchor-checked
    # grading. A vague rubric (v1-shaped: no anchors, no evidence
    # requirement) gets what an under-specified judge actually produces —
    # a shallow "does this look like an answer" pass that misses invented
    # numbers and penalises correct refusals. That gap, not a hand-picked
    # kappa, is the whole point of calibrating before trusting a judge.
    anchored = "declining to guess" in rubric_text.lower() and "quote" in rubric_text.lower()

    dont_know = ("don't have that information" in answer_text.lower()) or ("لا تتوفر لدي هذه المعلومة" in answer_text)

    if not anchored:
        if dont_know:
            return {"score": 0.0, "evidence": "does not look like it answered the question"}
        if not answer_text.strip():
            return {"score": 0.0, "evidence": "no answer given"}
        return {"score": 1.0, "evidence": "reads like a complete, on-topic answer"}

    if unsupported_amounts(answer_text, context_text):
        return {"score": 0.0, "evidence": "states an amount absent from the context"}
    if dont_know:
        return {"score": 1.0, "evidence": "correctly declines to guess when the context has no answer"}
    if is_grounded(answer_text, context_text):
        return {"score": 1.0, "evidence": "every stated amount appears in the context"}
    return {"score": 0.5, "evidence": "catalog-adjacent but not a precise quote"}


def _tool_result_reply(request: LLMRequest, language: str) -> LLMResponse | None:
    """If the last message is a tool result, summarise it instead of calling
    the tool again — the loop advances only if the model treats a result as
    an answer, exactly like a real model would after seeing ``role: tool``."""
    if not request.messages or request.messages[-1].role != "tool":
        return None
    last = request.messages[-1]
    try:
        payload = json.loads(last.content or "{}")
    except json.JSONDecodeError:
        payload = {}
    ar = language == "ar"

    if "error" in payload:
        hint = payload.get("hint", "")
        text = (f"عذراً، {hint}" if ar else f"Sorry — {hint}") if hint else (
            "عذراً، لم أستطع إتمام ذلك." if ar else "Sorry, I couldn't complete that."
        )
        return LLMResponse(text=text, finish_reason="stop", model_id=request.model_alias, usage=Usage(input_tokens=90, output_tokens=20), route="fake")

    if last.name == "check_order_status" and "status" in payload:
        status = payload["status_ar"] if ar else payload["status"]
        ref = payload.get("reference", "")
        text = f"طلبك {ref} حالته الآن: {status}." if ar else f"Your order {ref} is currently: {status}."
        return LLMResponse(text=text, finish_reason="stop", model_id=request.model_alias, usage=Usage(input_tokens=110, output_tokens=30), route="fake")

    if last.name == "create_return_request" and "case_id" in payload:
        case_id = payload["case_id"]
        text = f"تم فتح طلب الاسترجاع، رقم المرجع {case_id}." if ar else f"Your return has been filed — case_id {case_id}."
        return LLMResponse(text=text, finish_reason="stop", model_id=request.model_alias, usage=Usage(input_tokens=110, output_tokens=30), route="fake")

    if last.name == "escalate_to_agent" and payload.get("handed_off"):
        text = "سأحوّلك الآن إلى موظف بشري." if ar else "I'm transferring you to a human agent now; they'll be with you shortly."
        return LLMResponse(text=text, finish_reason="stop", model_id=request.model_alias, usage=Usage(input_tokens=90, output_tokens=20), route="fake")

    text = "تم." if ar else "Done."
    return LLMResponse(text=text, finish_reason="stop", model_id=request.model_alias, usage=Usage(input_tokens=80, output_tokens=10), route="fake")


def _service_reply(request: LLMRequest, text: str, user_text: str) -> LLMResponse:
    tool_reply = _tool_result_reply(request, _language(text))
    if tool_reply is not None:
        return tool_reply

    order_ref = _ORDER_REF.search(user_text)
    date = _DATE.search(user_text)
    resolution = "exchange" if _contains_any(user_text, _EXCHANGE_WORDS) else ("refund" if _contains_any(user_text, _RETURN_WORDS) else None)
    city = _city(user_text)
    confirmed = _contains_any(text, _CONFIRM_WORDS) or _contains_any(user_text, _CONFIRM_WORDS)

    if _contains_any(text, _ESCALATE_WORDS):
        return LLMResponse(
            tool_calls=[ToolCall(id="call_1", name="escalate_to_agent", arguments=json.dumps({"reason": text[:150]}, ensure_ascii=False))],
            finish_reason="tool_calls", model_id=request.model_alias, usage=Usage(input_tokens=140, output_tokens=30), route="fake",
        )

    if _contains_any(text, _RETURN_WORDS + _EXCHANGE_WORDS):
        if order_ref and resolution and city != "unknown" and date and confirmed:
            category = _category(user_text)
            if category not in ("electronics", "home_goods", "fashion"):
                from retail_support.tools.services import ORDERS

                order = ORDERS.get(order_ref.group(0).upper())
                category = order.category if order is not None else "electronics"
            args = {
                "order_reference": order_ref.group(0).upper(), "category": category,
                "resolution": resolution, "dropoff_city": city, "dropoff_date": date.group(0),
            }
            return LLMResponse(
                tool_calls=[ToolCall(id="call_1", name="create_return_request", arguments=json.dumps(args, ensure_ascii=False))],
                finish_reason="tool_calls", model_id=request.model_alias, usage=Usage(input_tokens=160, output_tokens=35), route="fake",
            )
        missing = []
        if not order_ref:
            missing.append("order reference")
        if not resolution:
            missing.append("resolution (refund or exchange)")
        if city == "unknown":
            missing.append("drop-off city")
        if not date:
            missing.append("date")
        ar = _language(text) == "ar"
        if not missing:
            # Everything is present but the customer has not explicitly said
            # yes — never file a return on an inferred confirmation.
            reply_text = (
                f"للتأكيد: استرجاع الطلب {order_ref.group(0).upper()}، {resolution}، "
                f"التسليم في {city} بتاريخ {date.group(0)} — هل تؤكد؟"
                if ar else
                f"To confirm: {resolution} for order {order_ref.group(0).upper()}, "
                f"drop-off in {city} on {date.group(0)} — shall I go ahead?"
            )
            return LLMResponse(text=reply_text, finish_reason="stop", model_id=request.model_alias, usage=Usage(input_tokens=130, output_tokens=30), route="fake")
        ask = ("رقم الطلب" if ar else "order reference") if "order reference" in missing else missing[0]
        reply_text = (
            f"بالتأكيد، ممكن تأكد لي {ask}؟" if ar else f"Happy to help — could you confirm the {ask}?"
        )
        return LLMResponse(text=reply_text, finish_reason="stop", model_id=request.model_alias, usage=Usage(input_tokens=120, output_tokens=25), route="fake")

    if _contains_any(text, _STATUS_WORDS) or order_ref:
        if order_ref:
            return LLMResponse(
                tool_calls=[ToolCall(id="call_1", name="check_order_status", arguments=json.dumps({"order_reference": order_ref.group(0).upper()}, ensure_ascii=False))],
                finish_reason="tool_calls", model_id=request.model_alias, usage=Usage(input_tokens=130, output_tokens=25), route="fake",
            )
        ar = _language(text) == "ar"
        return LLMResponse(
            text=("ممكن رقم الطلب (reference)؟" if ar else "Could you give me the order reference?"),
            finish_reason="stop", model_id=request.model_alias, usage=Usage(input_tokens=100, output_tokens=20), route="fake",
        )

    ar = _language(text) == "ar"
    return LLMResponse(
        text=("تقدر تعطيني رقم الطلب عشان أساعدك؟" if ar else "Sure — what's your order, and what would you like to do with it?"),
        finish_reason="stop", model_id=request.model_alias, usage=Usage(input_tokens=100, output_tokens=20), route="fake",
    )


_SEEN_PREFIXES: set[str] = set()


def _prefix_hash(request: LLMRequest) -> str | None:
    n = request.cache_prefix_messages
    if not n:
        return None
    import hashlib

    prefix = "".join(m.content for m in request.messages[:n])
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def _apply_cache_simulation(request: LLMRequest, response: LLMResponse) -> LLMResponse:
    """Simulate what a real provider's prompt cache would report.

    The stable prefix discipline (Module 6 §3) only pays off on the SECOND and
    later request that sends a byte-identical prefix — the first one is
    always a cold write. This mirrors that shape deterministically: a
    module-level set of prefix hashes stands in for the provider's cache
    state across the run.
    """
    prefix_hash = _prefix_hash(request)
    if prefix_hash is not None:
        if prefix_hash in _SEEN_PREFIXES:
            response.usage.cached_input_tokens = int(response.usage.input_tokens * 0.85)
        else:
            _SEEN_PREFIXES.add(prefix_hash)
    return response


def smart_default(request: LLMRequest) -> LLMResponse:
    text = _last_user_text(request)
    system_prompt = next((m.content for m in request.messages if m.role == "system"), "")

    if request.response_format:
        name = (request.response_format.get("json_schema") or {}).get("name", "")
        if name == "route_verdict":
            payload = _route_verdict(text)
        elif name == "guard_verdict":
            payload = _guard_verdict(text, system_prompt)
        elif name == "return_case":
            payload = _extract_return_case(text)
        elif name == "judge_verdict":
            payload = _judge_verdict(text)
        else:
            payload = {}
        response = LLMResponse(
            text=json.dumps(payload, ensure_ascii=False), finish_reason="stop",
            model_id=request.model_alias, usage=Usage(input_tokens=150, output_tokens=40), route="fake",
        )
        return _apply_cache_simulation(request, response)

    if request.tools:
        response = _service_reply(request, text, _user_text(request))
        return _apply_cache_simulation(request, response)

    language = _language(text)
    response = LLMResponse(
        text=_faq_answer(text, language), finish_reason="stop",
        model_id=request.model_alias, usage=Usage(input_tokens=160, output_tokens=60), route="fake",
    )
    return _apply_cache_simulation(request, response)
