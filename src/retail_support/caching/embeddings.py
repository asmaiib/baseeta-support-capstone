"""A small, local, deterministic embedding — enough to demonstrate semantic caching.

It hashes character n-grams into a fixed-width vector and normalises. That is not
a sentence embedding and it is not pretending to be one: it has no idea that
"renew" and "cancel" are opposites, which is precisely the failure mode the
near-miss suite exists to catch, and it catches it here for the same reason a real
embedding does — surface similarity is not meaning.

Swap in a real embedding model and the lesson does not change: **a semantic cache
is a model, and you evaluate it like one.** The threshold that is safe for your
traffic is a number you measure, never a number you guess.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from functools import lru_cache

DIMENSIONS = 256
NGRAM = 3

_PUNCT = re.compile(r"[^\w؀-ۿ]+")
#: Arabic diacritics and the tatweel: two spellings of one word should not be two
#: different cache keys.
_DIACRITICS = re.compile("[" + "".join(chr(c) for c in range(0x064B, 0x0653)) + "ـ]")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower())
    text = _DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    return " " + _PUNCT.sub(" ", text).strip() + " "


@lru_cache(maxsize=4096)
def embed(text: str) -> tuple[float, ...]:
    vector = [0.0] * DIMENSIONS
    prepared = normalise(text)
    for i in range(max(len(prepared) - NGRAM + 1, 1)):
        gram = prepared[i : i + NGRAM]
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % DIMENSIONS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return tuple(v / norm for v in vector)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def similarity(a: str, b: str) -> float:
    return cosine(embed(a), embed(b))
