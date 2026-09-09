"""The course gateway's *simulator*. Not a model — a deterministic stand-in.

Read this before you trust a number that came out of it.

Why it exists
-------------
Every lab, test and eval run in this course must work on a laptop with no key, no
network and no GPU, and every drill (the 429 storm, the provider outage, the
prompt-cache proof) must be reproducible on demand rather than depending on a
provider having a bad day at the right moment. So the course ships a gateway that
speaks both wire dialects — ``POST /v1/chat/completions`` and ``POST /v1/messages``
— and answers from rules instead of weights.

What is real and what is simulated
----------------------------------
Real: the wire contract, ``usage`` accounting, ``finish_reason`` control flow, tool
calls, structured outputs, streaming frames, prompt-cache hits keyed on a genuinely
byte-stable prefix, HTTP error taxonomy, and the fact that answers are **grounded in
the service directory that arrives in the prompt** — if a fact is not in the
directory the simulator cannot produce it, exactly like a well-guarded model should
not.

Simulated: quality. Model "tiers" differ by rules, not by capability:

* ``course-flagship`` follows the prompt's don't-know rule and extracts cleanly;
* ``course-small`` is cheaper and deterministically worse — it guesses a fee for
  out-of-directory questions about a third of the time and fumbles more schemas;
* ``retail_support-onprem`` stands in for a 7B open-weight model: worse again, and worse
  on Arabic specifically.

Those degradations are a hash of the input, not a random draw, so a run is
reproducible and a regression gate that fires, fires for a reason you can point at.

The honest caveat, which belongs in your evaluation report
----------------------------------------------------------
Numbers measured against this gateway are numbers about *this simulator*. They are
correct as measurements of the harness, the guards, the meter and the gate — the
things this course actually teaches — and they are not evidence about any real
model. Point a route at a real provider (two env vars, no code change) and re-run
the same harness when you want evidence about a model. That substitution being a
config edit is the entire argument of Module 1.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Model tiers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    name: str
    #: chance an out-of-directory question gets an invented fee instead of "I don't know"
    guess_rate: float
    #: chance a structured extraction fails validation on the first attempt
    schema_fail_rate: float
    #: extra failure chance on Arabic input (the gap the repair loop closes)
    schema_fail_rate_ar_extra: float
    #: chance the model fumbles a tool contract (wrong/missing argument)
    tool_fumble_rate: float
    #: output verbosity multiplier — a proxy for cost, and for snapshot drift
    verbosity: float = 1.0
    latency_base_ms: float = 250.0
    latency_per_token_ms: float = 1.5
    prompt_ms_per_1k: float = 90.0


TIERS: dict[str, Tier] = {
    "course-flagship": Tier("course-flagship", 0.0, 0.06, 0.03, 0.02, 1.0, 260, 1.6, 95),
    "course-anthropic": Tier("course-anthropic", 0.0, 0.05, 0.03, 0.02, 1.05, 300, 1.7, 100),
    "course-small": Tier("course-small", 0.34, 0.16, 0.08, 0.14, 0.8, 120, 0.7, 45),
    "retail_support-onprem": Tier("retail_support-onprem", 0.45, 0.20, 0.14, 0.22, 0.9, 360, 1.1, 130),
}
DEFAULT_TIER = TIERS["course-flagship"]


def tier_for(model: str) -> Tier:
    for key, tier in TIERS.items():
        if model.startswith(key):
            return tier
    return DEFAULT_TIER


def _roll(*parts: str) -> float:
    """Deterministic pseudo-random in [0, 1) — same input, same verdict, forever."""
    digest = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()  # noqa: S324
    return int(digest[:8], 16) / 0xFFFFFFFF


# --------------------------------------------------------------------------
# Language and light NLU
# --------------------------------------------------------------------------

ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")

CITIES = {
    "Riyadh": ["riyadh", "الرياض"],
    "Jeddah": ["jeddah", "jedda", "جدة"],
    "Makkah": ["makkah", "mecca", "مكة"],
    "Dammam": ["dammam", "الدمام"],
    "Madinah": ["madinah", "medina", "المدينة"],
    "Abha": ["abha", "أبها", "ابها"],
    "Tabuk": ["tabuk", "تبوك"],
    "Buraidah": ["buraidah", "بريدة"],
}

URGENT_WORDS = ["urgent", "asap", "immediately", "عاجل", "بسرعة", "ضروري"]
EMERGENCY_WORDS = ["emergency", "طارئ", "حالة طارئة"]
HUMAN_WORDS = [
    "human",
    "agent",
    "representative",
    "complaint",
    "a person",
    "to a person",
    "real person",
    "someone who can",
    "speak to somebody",
    "موظف",
    "ممثل",
    "شكوى",
    "أشتكي",
    "بشر",
    "شخص",
    "مسؤول",
]
CRISIS_WORDS = ["kill myself", "suicide", "hurt myself", "أنهي حياتي", "الانتحار", "أؤذي نفسي"]

REFERENCE = re.compile(r"\b[A-Z]{2}\d{8}\b")
NATIONAL_ID = re.compile(r"\b[12]\d{9}\b")
PHONE = re.compile(r"(?:\+?966|0)5\d{8}\b")
DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


def detect_language(text: str) -> str:
    ar = len(ARABIC.findall(text))
    en = len(LATIN.findall(text))
    if ar and en:
        ratio = ar / max(ar + en, 1)
        if 0.2 < ratio < 0.8:
            return "mixed"
        return "ar" if ratio >= 0.8 else "en"
    return "ar" if ar else "en"


def detect_city(text: str) -> str:
    low = text.lower()
    for canonical, forms in CITIES.items():
        if any(f in low for f in forms):
            return canonical
    return "unknown"


def _contains_any(text: str, words: list[str]) -> bool:
    low = text.lower()
    return any(w.lower() in low for w in words)


# --------------------------------------------------------------------------
# Parsing the directory *out of the prompt* — groundedness for real
# --------------------------------------------------------------------------

ENTRY_HEAD = re.compile(r"^###\s+(\S+)\s+—\s+(.+)$")


@dataclass
class DirEntry:
    id: str
    title: str
    service_type: str = "other"
    fee: str = ""
    processing_time: str = ""
    documents: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


def parse_directory(text: str) -> list[DirEntry]:
    entries: list[DirEntry] = []
    current: DirEntry | None = None
    for line in text.splitlines():
        line = line.rstrip()
        head = ENTRY_HEAD.match(line)
        if head:
            current = DirEntry(id=head.group(1), title=head.group(2).strip())
            entries.append(current)
            continue
        if current is None or not line.startswith("- "):
            continue
        key, _, value = line[2:].partition(":")
        value = value.strip()
        key = key.strip()
        if key == "service_type":
            current.service_type = value
        elif key == "fee":
            current.fee = value
        elif key == "processing_time":
            current.processing_time = value
        elif key == "documents":
            current.documents = [d.strip() for d in value.split(";") if d.strip()]
        elif key == "steps":
            current.steps = [s.strip() for s in value.split("|") if s.strip()]
        elif key == "keywords":
            current.keywords = [k.strip().lower() for k in value.split(",") if k.strip()]
    return entries


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return match.group(1).strip() if match else ""


def _fold(token: str) -> str:
    """Fold the spelling variants that would otherwise split one word into three."""
    token = token.replace(chr(0x0623), chr(0x0627)).replace(chr(0x0625), chr(0x0627))
    token = token.replace(chr(0x0622), chr(0x0627)).replace(chr(0x0649), chr(0x064A))
    token = token.replace(chr(0x0629), chr(0x0647))
    if token.startswith(chr(0x0627) + chr(0x0644)) and len(token) > 4:
        token = token[2:]  # the definite article is not a distinguishing feature
    return token


def tokenise(text: str) -> list[str]:
    tokens = re.sub(r"[^\w]+", " ", text.lower()).split()
    return [_fold(t) for t in tokens if len(t) >= 3]


STOPWORDS = {
    "the", "for", "and", "how", "what", "with", "you", "your", "can", "need",
    "want", "does", "are", "get", "från", "من", "الي", "علي", "هي", "هل", "كيف",
    "ماذا", "اريد", "هذا", "التي", "الذي", "عن", "في", "ما", "لي",
}


def _entry_tokens(entry: DirEntry) -> set[str]:
    bag: set[str] = set()
    for keyword in entry.keywords:
        bag.update(tokenise(keyword))
    bag.update(tokenise(entry.title))
    return {t for t in bag if t not in STOPWORDS}


CLITICS = "لبكوف"  # la-, bi-, ka-, wa-, fa- : Arabic writes them joined


def expand(tokens: list[str]) -> set[str]:
    """Arabic writes several prepositions joined to the next word. Index both
    forms so that "lirukhsa" and "rukhsa" are the same evidence."""
    out: set[str] = set()
    for token in tokens:
        out.add(token)
        if token[0] in CLITICS and len(token) > 3:
            out.add(token[1:])
    return out


def match_entry(message: str, entries: list[DirEntry]) -> tuple[DirEntry | None, float]:
    """Weighted token overlap. Crude on purpose: retrieval is SDA-AIE-214's course.

    Two things make it behave. Distinctive tokens count for more than common ones,
    or "renewing a driving licence" matches the *commercial* renewal entry because
    every entry shares the word "renew". And a single shared token is never enough
    on its own — that rule is what keeps "what is the fee for a falconry permit?"
    away from the building-permit entry, and keeps the don't-know answer honest.
    """
    if not entries:
        return None, 0.0
    bags = [_entry_tokens(e) for e in entries]
    document_frequency: dict[str, int] = {}
    for bag in bags:
        for token in bag:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    message_tokens = expand([t for t in tokenise(message) if t not in STOPWORDS])
    if not message_tokens:
        return None, 0.0

    best: DirEntry | None = None
    best_score = 0.0
    for entry, bag in zip(entries, bags, strict=True):
        matched = message_tokens & bag
        score = sum(1.0 / document_frequency[t] for t in matched)
        phrases = 0
        for keyword in entry.keywords:
            keyword_tokens = {t for t in tokenise(keyword) if t not in STOPWORDS}
            if len(keyword_tokens) >= 2 and keyword_tokens <= message_tokens:
                phrases += 1
        score += phrases
        if len(matched) < 2 and phrases == 0:
            continue  # one shared word is a coincidence, not a match
        if score > best_score:
            best, best_score = entry, score
    return (best, best_score) if best_score >= 0.8 else (None, best_score)


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass
class BrainResult:
    text: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"


DONT_KNOW = {
    "en": (
        "I don't have that information in the service directory, so I won't guess. "
        "Please check with the service centre: {centre}"
    ),
    "ar": (
        "لا تتوفر لدي هذه المعلومة في دليل الخدمات، ولن أخمّن. "
        "يرجى مراجعة مركز الخدمة: {centre}"
    ),
}
INVENTED_FEES_EN = ["SAR 150", "SAR 250", "SAR 75"]
INVENTED_FEES_AR = ["١٥٠ ريالاً", "٢٥٠ ريالاً", "٧٥ ريالاً"]

# The prompt's don't-know rule, in both languages. If the served prompt version
# does not carry it, the simulator guesses — which is how Lab 5's seeded prompt
# regression stops being a story and starts being a measurement.
DONT_KNOW_RULE_MARKERS = [
    "say you do not know",
    "say you don't know",
    "لا تعرف",
]


def _centre(directory_text: str, language: str) -> str:
    match = re.search(r"service_centre:\s*(.+)", directory_text)
    if match:
        return match.group(1).strip()
    return "199" if language == "ar" else "the service centre (199)"


# --------------------------------------------------------------------------
# Task 1 — the FAQ answer, grounded in the directory that arrived in the prompt
# --------------------------------------------------------------------------


def answer_faq(system: str, user_text: str, model: str) -> BrainResult:
    tier = tier_for(model)
    directory_text = extract_tag(system, "service_directory") or system
    entries = parse_directory(directory_text)
    message = (
        extract_tag(system, "customer_message")
        or extract_tag(user_text, "customer_message")
        or user_text
    )
    language = detect_language(message)
    language = "ar" if language in ("ar", "mixed") else "en"
    entry, _score = match_entry(message, entries)
    centre = _centre(directory_text, language)

    if entry is None:
        rule_present = any(m in system.lower() for m in DONT_KNOW_RULE_MARKERS)
        guess_rate = tier.guess_rate if rule_present else max(tier.guess_rate, 0.6)
        if _roll(model, message, "guess") < guess_rate:
            fee = (INVENTED_FEES_AR if language == "ar" else INVENTED_FEES_EN)[
                int(_roll(message, "fee") * 3) % 3
            ]
            if language == "ar":
                text = (
                    f"بالطبع، رسوم هذه الخدمة {fee} وتستغرق عادةً ثلاثة أيام عمل. "
                    "يمكنك إتمامها من البوابة مباشرة."
                )
            else:
                text = (
                    f"Of course — the fee for that service is {fee} and it usually "
                    "takes about three working days. You can complete it in the portal."
                )
            return BrainResult(text=text)
        return BrainResult(text=DONT_KNOW[language].format(centre=centre))

    if language == "ar":
        parts = [
            f"بخصوص {entry.title}:",
            f"- الرسوم: {entry.fee}",
            f"- المدة: {entry.processing_time}",
            "- المستندات المطلوبة: " + "، ".join(entry.documents),
            "- الخطوات:",
        ]
        parts += [f"  {i}. {s}" for i, s in enumerate(entry.steps, start=1)]
        parts.append(f"إذا احتجت مساعدة إضافية يمكنك مراجعة {centre}")
    else:
        parts = [
            f"About {entry.title}:",
            f"- Fee: {entry.fee}",
            f"- Processing time: {entry.processing_time}",
            "- Documents required: " + "; ".join(entry.documents),
            "- Steps:",
        ]
        parts += [f"  {i}. {s}" for i, s in enumerate(entry.steps, start=1)]
        parts.append(f"If you need more help you can contact {centre}")
    text = "\n".join(parts)
    if tier.verbosity > 1.0:
        text += (
            "\n\nI hope this is helpful. Let me know if you would like me to walk "
            "through any of these steps in more detail."
        )
    return BrainResult(text=text)


# --------------------------------------------------------------------------
# Task 2 — structured ticket extraction
# --------------------------------------------------------------------------

SERVICE_TYPE_HINTS = {
    "commercial_licence": ["commercial", "cr", "business", "trade", "سجل تجاري", "رخصة تجارية", "مؤسسة"],
    "civil_records": ["identity", "id card", "birth", "civil", "family", "هوية", "ميلاد", "أحوال", "الأحوال المدنية"],
    "traffic_services": ["driving", "vehicle", "car", "traffic", "قيادة", "مركبة", "سيارة", "مرور"],
    "municipal_permits": ["building permit", "municipal", "shop", "restaurant", "بناء", "بلدية", "محل", "مطعم"],
}


def _service_type(message: str, entry: DirEntry | None) -> str:
    if entry is not None and entry.service_type in {
        "commercial_licence",
        "civil_records",
        "traffic_services",
        "municipal_permits",
    }:
        return entry.service_type
    low = message.lower()
    for service_type, hints in SERVICE_TYPE_HINTS.items():
        if any(h in low for h in hints):
            return service_type
    return "other"


def _full_name(message: str) -> str:
    patterns = [
        r"my name is ([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})",
        r"[Ii](?:'m| am) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})",
        r"اسمي ([؀-ۿ]+(?: [؀-ۿ]+){0,3})",
        r"أنا ([؀-ۿ]+(?: [؀-ۿ]+){0,2})(?:،|\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1).strip()
    return ""


def extract_ticket(system: str, user_text: str, model: str, repairing: bool) -> BrainResult:
    tier = tier_for(model)
    directory_text = extract_tag(system, "service_directory")
    entries = parse_directory(directory_text) if directory_text else []
    message = extract_tag(user_text, "customer_message") or user_text
    language = detect_language(message)
    entry, _ = match_entry(message, entries) if entries else (None, 0.0)

    national_id = NATIONAL_ID.search(message)
    phone = PHONE.search(message)
    name = _full_name(message)
    city = detect_city(message)
    urgency = (
        "emergency"
        if _contains_any(message, EMERGENCY_WORDS)
        else "urgent"
        if _contains_any(message, URGENT_WORDS)
        else "routine"
    )
    needs_human = _contains_any(message, HUMAN_WORDS) or _contains_any(message, CRISIS_WORDS)
    service_type = _service_type(message, entry)
    summary = (
        f"Customer asks about {entry.title.lower()}" if entry else "Customer asks about a government service"
    )
    if city != "unknown":
        summary += f" in {city}"
    summary += "."

    ticket: dict[str, Any] = {
        "service_type": service_type,
        "summary_en": summary,
        "city": city,
        "urgency": urgency,
        "language": language,
        "applicant": {
            "full_name": name or ("Unnamed customer" if language == "en" else "مواطن لم يذكر اسمه"),
            "national_id": national_id.group() if national_id else None,
            "phone": phone.group() if phone else None,
        },
        "needs_human": needs_human,
    }

    if repairing:
        # One repair, not an unbounded loop — and repair is not magic. A small
        # share of cases stay broken, which is what puts the escalation path on
        # the corpus report instead of only in the design document.
        stubborn = 0.25 if tier.name == "retail_support-onprem" else 0.12
        if _roll(model, message, "repairfail") < stubborn:
            ticket["urgency"] = "high"
    else:
        fail_rate = tier.schema_fail_rate + (
            tier.schema_fail_rate_ar_extra if language in ("ar", "mixed") else 0.0
        )
        roll = _roll(model, message, "schema")
        if roll < fail_rate:
            # Three realistic first-try failures, in rotation. Each one is a
            # *semantic* violation the JSON Schema cannot catch — pydantic's job.
            mode = int(_roll(message, "failmode") * 3) % 3
            if mode == 0:
                ticket["city"] = "Al Khobar"  # not in the enum
            elif mode == 1:
                ticket["applicant"]["national_id"] = "9" + "12345678"  # wrong leading digit
            else:
                ticket["urgency"] = "high"  # not in the enum
        elif tier.name == "retail_support-onprem" and _roll(model, message, "invent") < 0.10:
            # The weaker model fills a blank rather than leaving it null.
            if ticket["city"] == "unknown":
                ticket["city"] = "Riyadh"
    return BrainResult(text=json.dumps(ticket, ensure_ascii=False))


# --------------------------------------------------------------------------
# Task 3 — guard classifier, router, judge (guards and gates are extraction too)
# --------------------------------------------------------------------------

# Phrase-level, not word-level. A classifier that fires on the bare word "ignore"
# blocks "I want to ignore the reminder emails" — one of the three traps planted
# in the legitimate corpus, and the reason Lab 4 task 5 exists.
INJECTION_HINTS = [
    "ignore all previous",
    "ignore your previous",
    "ignore the previous",
    "ignore previous instructions",
    "disregard the above",
    "disregard all previous",
    "system prompt",
    "developer message",
    "reveal your",
    "print your instructions",
    "show me your instructions",
    "your initial prompt",
    "your configuration",
    "you are now",
    "you are no longer",
    "new instructions:",
    "repeat everything above",
    "act as the system",
    "تجاهل كل التعليمات",
    "تجاهل التعليمات",
    "تجاهل كل الأوامر",
    "تعليمات النظام",
    "أظهر تعليماتك",
    "اطبع التعليمات",
    "تعليماتك الأصلية",
    "البرومبت",
    "أنت الآن",
    "تعليمات جديدة",
    "قواعدك الداخلية",
    "تصرف كمدير",
    "كرر كل ما",
    "إعداداتك",
]
OFF_SCOPE_HINTS = [
    "medicine",
    "diagnos",
    "symptom",
    "election",
    "political",
    "homework",
    "stock",
    "bitcoin",
    "دواء",
    "تشخيص",
    "أعراض",
    "سياس",
    "واجب",
    "أسهم",
]


def classify_guard(user_text: str, model: str) -> BrainResult:
    message = extract_tag(user_text, "customer_message") or user_text
    low = message.lower()
    if _contains_any(message, CRISIS_WORDS):
        category = "crisis"
    elif any(h in low for h in INJECTION_HINTS):
        category = "injection_attempt"
    elif any(h in low for h in OFF_SCOPE_HINTS):
        category = "off_scope"
    else:
        category = "ok"
    # A small classifier is not a perfect classifier. Deterministic, and rare.
    if tier_for(model).name in ("course-small", "retail_support-onprem") and category == "injection_attempt":
        if _roll(model, message, "guardmiss") < 0.06:
            category = "ok"
    return BrainResult(text=json.dumps({"category": category}, ensure_ascii=False))


# Action phrasing, not topic words. "How far ahead can I book an appointment?" is
# a question about a service and belongs on the FAQ route; "book me an appointment"
# is a request to act. A router that keys on the noun sends the first one to the
# expensive handler and then wonders where the money went.
SERVICE_INTENT_HINTS = [
    "book me",
    "please book",
    "i want to book",
    "i would like to book",
    "book an appointment in",
    "book a civil records appointment",
    "book a traffic services appointment",
    "i need an appointment",
    "cancel my",
    "reschedule",
    "check my application",
    "status of my",
    "status of application",
    "my application",
    "check application",
    "احجز",
    "أريد حجز",
    "احجز لي",
    "أكّد الحجز",
    "أكد الحجز",
    "أؤكد الحجز",
    "أبغى موعد",
    "أحتاج موعد",
    "ألغِ",
    "ألغي موعد",
    "حالة الطلب",
    "حالة طلبي",
    "طلبي رقم",
    "رقم الطلب",
    "استعلم",
    "الاستعلام عن طلب",
    "وين وصل",
]


def classify_route(user_text: str, model: str) -> BrainResult:
    message = extract_tag(user_text, "customer_message") or user_text
    if _contains_any(message, HUMAN_WORDS) or _contains_any(message, CRISIS_WORDS):
        intent = "escalate"
    elif _contains_any(message, SERVICE_INTENT_HINTS) or REFERENCE.search(message):
        intent = "service"
    else:
        intent = "faq"
    if tier_for(model).name == "retail_support-onprem" and _roll(model, message, "misroute") < 0.05:
        intent = "faq" if intent == "service" else "service"
    return BrainResult(text=json.dumps({"intent": intent}, ensure_ascii=False))


CURRENCY = re.compile(r"(SAR\s*[\d,]+|[\d,]+\s*(?:ريال|ريالاً|رياﻻ))")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _fees_in(text: str) -> set[str]:
    out = set()
    for match in CURRENCY.finditer(text.translate(ARABIC_DIGITS)):
        digits = re.sub(r"\D", "", match.group())
        if digits:
            out.add(digits)
    return out


def judge(system: str, user_text: str, model: str) -> BrainResult:
    """Grades ONE dimension: groundedness. The rubric's quality decides the judge's.

    A vague rubric produces a noisy judge (kappa around 0.4). A rubric that demands
    an evidence quote and states the don't-know clause produces a judge that agrees
    with the humans (kappa above 0.6). Fix the rubric, not the humans.
    """
    rubric = extract_tag(user_text, "rubric") or extract_tag(system, "rubric") or system
    directory_text = extract_tag(user_text, "context") or extract_tag(user_text, "service_directory")
    answer = extract_tag(user_text, "answer") or user_text

    answer_fees = _fees_in(answer)
    directory_fees = _fees_in(directory_text)
    ungrounded = answer_fees - directory_fees
    says_dont_know = any(
        m in answer.lower()
        for m in ["don't have that information", "do not have that information", "لا تتوفر لدي"]
    )

    if ungrounded:
        score, evidence = 0.0, (
            f"The answer states a fee ({sorted(ungrounded)[0]}) that appears nowhere in the directory."
        )
    elif says_dont_know:
        score, evidence = 1.0, "The answer declines to guess and points to the service centre."
    elif answer_fees & directory_fees:
        score, evidence = 1.0, "Every fee stated in the answer appears in the directory."
    else:
        score, evidence = 0.5, "No fee claim to check; claims are directory-shaped but imprecise."

    # What makes a rubric an instrument rather than a mood: it demands evidence,
    # and it states what to do with an answer that correctly declines to answer.
    # A rubric missing either one produces a judge that agrees with humans about
    # two thirds of the time, which is worse than useless because it looks like a
    # measurement. This check is the simulator's stand-in for that difference.
    low_rubric = rubric.lower()
    demands_evidence = "evidence" in low_rubric or "quote" in low_rubric
    handles_refusal = any(
        phrase in low_rubric
        for phrase in ("does not know", "do not know", "don't know", "declin", "لا يعرف")
    )
    sharp = demands_evidence and handles_refusal
    if not sharp:
        # An unanchored rubric is an unreliable instrument. This is the whole
        # lesson of Lab 5 task 3, and it has to be *measurable* to land.
        roll = _roll(model, answer, "judgenoise")
        if roll < 0.34:
            score = 1.0 if score < 1.0 else 0.5
            evidence = "The answer reads well."
    return BrainResult(text=json.dumps({"score": score, "evidence": evidence}, ensure_ascii=False))


# --------------------------------------------------------------------------
# Task 4 — the tool loop
# --------------------------------------------------------------------------

BOOK_CONFIRM_WORDS = ["confirm", "yes", "go ahead", "please book", "نعم", "أكد", "أكّد", "احجز"]


def decide_tools(messages: list[dict], tools: list[dict], model: str) -> BrainResult:
    tier = tier_for(model)
    by_name = {}
    for t in tools:
        fn = t.get("function", t)
        by_name[fn["name"]] = fn

    tool_results = [m for m in messages if m.get("role") == "tool"]
    last_user = next(
        (m for m in reversed(messages) if m.get("role") == "user" and isinstance(m.get("content"), str)),
        {"content": ""},
    )
    user_text = last_user.get("content") or ""
    message = extract_tag(user_text, "customer_message") or user_text
    language = "ar" if detect_language(message) in ("ar", "mixed") else "en"

    # An over-broad description routes traffic that should never reach the tool.
    # "Descriptions route" — the one-line rule from the sim-greedy-tools branch.
    for name, fn in by_name.items():
        if "any question" in (fn.get("description") or "").lower() and not tool_results:
            return BrainResult(
                tool_calls=[{"name": name, "arguments": {"reference": "CR00000000"}}],
                finish_reason="tool_calls",
            )

    if tool_results:
        last = tool_results[-1]
        try:
            payload = json.loads(last.get("content") or "{}")
        except json.JSONDecodeError:
            payload = {}
        if "error" in payload:
            hint = payload.get("hint", "")
            if language == "ar":
                text = "لم أتمكن من إتمام العملية. " + (
                    hint or "هل يمكنك التأكد من البيانات والمحاولة مرة أخرى؟"
                )
            else:
                text = "I couldn't complete that. " + (
                    hint or "Could you confirm the details and try again?"
                )
            return BrainResult(text=text)
        if "status" in payload:
            if language == "ar":
                text = (
                    f"حالة الطلب {payload.get('reference', '')}: {payload.get('status_ar', payload['status'])}. "
                    f"آخر تحديث: {payload.get('updated', '')}."
                )
            else:
                text = (
                    f"Application {payload.get('reference', '')} is currently "
                    f"'{payload['status']}', last updated {payload.get('updated', '')}."
                )
            return BrainResult(text=text)
        if "confirmation" in payload:
            if language == "ar":
                text = (
                    f"تم الحجز. رقم التأكيد {payload['confirmation']} "
                    f"في {payload.get('city', '')} بتاريخ {payload.get('date', '')}."
                )
            else:
                text = (
                    f"Booked. Your confirmation number is {payload['confirmation']} "
                    f"in {payload.get('city', '')} on {payload.get('date', '')}."
                )
            return BrainResult(text=text)
        if "handed_off" in payload:
            text = (
                "تم تحويلك إلى موظف خدمة، سيصلك رد قريباً."
                if language == "ar"
                else "I've transferred you to a human agent; they'll be with you shortly."
            )
            return BrainResult(text=text)
        return BrainResult(text=json.dumps(payload, ensure_ascii=False))

    if (_contains_any(message, HUMAN_WORDS) or _contains_any(message, CRISIS_WORDS)) and (
        "escalate_to_agent" in by_name
    ):
        reason = "customer asked for a human agent"
        if _contains_any(message, CRISIS_WORDS):
            reason = "customer may be in distress"
        return BrainResult(
            tool_calls=[{"name": "escalate_to_agent", "arguments": {"reason": reason}}],
            finish_reason="tool_calls",
        )

    reference = REFERENCE.search(message)
    if reference and "check_application_status" in by_name:
        args = {"reference": reference.group()}
        if _roll(model, message, "toolfumble") < tier.tool_fumble_rate:
            args = {"reference": reference.group().lower()}  # violates the pattern
        return BrainResult(
            tool_calls=[{"name": "check_application_status", "arguments": args}],
            finish_reason="tool_calls",
        )

    wants_status = _contains_any(message, ["status", "حالة الطلب", "طلبي", "وين وصل"])
    if wants_status and "check_application_status" in by_name and not reference:
        text = (
            "بالتأكيد — ما رقم الطلب؟ يتكوّن من حرفين وثمانية أرقام، مثل CR12345678."
            if language == "ar"
            else "Happy to check — what is the reference number? It is two letters and eight digits, e.g. CR12345678."
        )
        return BrainResult(text=text)

    wants_booking = _contains_any(message, ["book", "appointment", "احجز", "حجز", "موعد"])
    if wants_booking and "book_appointment" in by_name:
        date = DATE.search(message)
        city = detect_city(message)
        confirmed = _contains_any(message, BOOK_CONFIRM_WORDS)
        if date and city != "unknown" and confirmed:
            service_type = _service_type(message, None)
            if service_type == "other":
                service_type = "civil_records"
            args = {"service_type": service_type, "city": city, "date": date.group()}
            if _roll(model, message, "bookfumble") < tier.tool_fumble_rate:
                args.pop("date")  # missing required argument
            return BrainResult(
                tool_calls=[{"name": "book_appointment", "arguments": args}],
                finish_reason="tool_calls",
            )
        missing = []
        if city == "unknown":
            missing.append("المدينة" if language == "ar" else "the city")
        if not date:
            missing.append("التاريخ (YYYY-MM-DD)" if language == "ar" else "the date (YYYY-MM-DD)")
        if not missing:
            text = (
                f"سأحجز موعداً في {city} بتاريخ {date.group()}. هل تؤكد؟"
                if language == "ar"
                else f"I'll book an appointment in {city} on {date.group()}. Shall I confirm?"
            )
        elif language == "ar":
            text = "لأحجز الموعد أحتاج " + " و".join(missing) + "."
        else:
            text = "To book the appointment I need " + " and ".join(missing) + "."
        return BrainResult(text=text)

    return BrainResult(text=None)  # caller falls through to the FAQ path
