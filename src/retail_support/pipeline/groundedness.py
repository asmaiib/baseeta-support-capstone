"""One deterministic groundedness check, used in three places on purpose.

It answers a single question: does this answer state a monetary amount that does
not appear in the trusted directory?

* the **eval harness** uses it as a blocking assert (safety-class claims are never
  a judge's to make);
* the **output guard** could use it, and a production system probably would;
* the **cascade** uses it as its escalation signal (Module 6 §5).

That last one is the interesting reuse. A cascade needs a signal that is cheap,
deterministic and actually correlated with being wrong — not the model's opinion
of its own answer, which is weak and sycophantic. "You quoted a number nobody gave
you" is exactly such a signal, and it costs one regex pass.
"""

from __future__ import annotations

import re

#: Amounts written either way round, in either script.
CURRENCY = re.compile(
    r"(?:SAR|ريال|ريالا|ريالاً|رياﻻ)\s*[\d٠-٩,]+"
    r"|[\d٠-٩,]+\s*(?:SAR|ريال|ريالا|ريالاً|رياﻻ)"
)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def amounts(text: str) -> set[str]:
    """Monetary amounts as bare digit strings, script-normalised."""
    found = set()
    for token in CURRENCY.findall(text.translate(ARABIC_DIGITS)):
        digits = re.sub(r"\D", "", token.translate(ARABIC_DIGITS))
        if digits:
            found.add(digits)
    return found


def unsupported_amounts(answer: str, directory: str) -> set[str]:
    """Amounts the answer states that the directory does not contain."""
    return amounts(answer) - amounts(directory)


def is_grounded(answer: str, directory: str) -> bool:
    return not unsupported_amounts(answer, directory)
