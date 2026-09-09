"""Same string, several token counts — all correct for their own model.

Used by the context-budget check and by the Module 6 cost forecast. The rule this
module exists to enforce: **count with the route's own tokenizer**, never with a
convenient one. ``sim-tokenizer-mismatch`` is what happens when you don't.

A finding the course measures rather than asserts: the
Arabic token premium is a property of the *tokenizer generation*, not of Arabic.
On ``cl100k_base`` (the GPT-4 / GPT-3.5 era) Arabic costs ~2.5x its English twin.
On ``o200k_base`` (GPT-4o era and after) the same corpus costs ~1.05x. Run
``python scripts/token_report.py`` and read your own numbers off the table before
you budget anything — including any figure printed in a course handout.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

#: Which encoding each configured model id actually uses. Guessing here is the bug.
MODEL_ENCODINGS: dict[str, str] = {
    "course-flagship": "o200k_base",
    "course-small": "o200k_base",
    "course-anthropic": "o200k_base",  # approximation: Anthropic's tokenizer is server-side
    "retail_support-onprem": "cl100k_base",  # stand-in for an open-weight SentencePiece vocab
}
DEFAULT_ENCODING = "o200k_base"


@lru_cache(maxsize=8)
def _encoding(name: str):
    return tiktoken.get_encoding(name)


def encoding_for_model(model_id: str) -> str:
    for key, enc in MODEL_ENCODINGS.items():
        if model_id.startswith(key):
            return enc
    return DEFAULT_ENCODING


def count(text: str, model_id: str = "course-flagship") -> int:
    """Token count for ``text`` as *this* model would see it."""
    return len(_encoding(encoding_for_model(model_id)).encode(text))


def count_with(text: str, encoding_name: str) -> int:
    return len(_encoding(encoding_name).encode(text))


def count_messages(messages, model_id: str = "course-flagship") -> int:
    """Rough request-side accounting: content plus a small per-message overhead.

    Providers add a handful of tokens per message for role framing. The exact
    number is provider- and version-specific, which is the point: this is a
    *budget* check, and a budget check that pretends to be exact is a lie with
    decimal places. Compare against ``usage.input_tokens`` from a real response.
    """
    per_message_overhead = 4
    total = 0
    for m in messages:
        total += count(getattr(m, "content", "") or "", model_id) + per_message_overhead
    return total


def arabic_premium(pairs: list[tuple[str, str]], encoding_name: str) -> float:
    """Arabic tokens ÷ English tokens over parallel sentence pairs."""
    en = sum(count_with(a, encoding_name) for a, _ in pairs)
    ar = sum(count_with(b, encoding_name) for _, b in pairs)
    return (ar / en) if en else 0.0
