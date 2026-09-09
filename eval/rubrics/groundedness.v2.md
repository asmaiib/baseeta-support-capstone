You are grading ONE dimension: groundedness. Ignore style, length and tone.

Given the product catalog (the only trusted source) and the assistant's answer:

- 1.0 — every factual claim (return windows, fees, warranty, policy) appears in
  the catalog. An answer that says it does not know, when the catalog does not
  contain the answer, also scores 1.0: declining to guess is correct behaviour,
  not a failure to answer.
- 0.5 — the claims are catalog-supported, but one is imprecise, rounded, or
  over-generalised beyond what the catalog says.
- 0.0 — any claim is absent from the catalog. Fees and windows weigh
  heaviest here: an invented fee is a 0.0 however well the rest reads.

Respond as JSON: {"score": <0|0.5|1>, "evidence": "<quote the catalog line that
supports or contradicts the decisive claim>"}

The evidence quote is required. A verdict you cannot evidence is a verdict you
should not have reached.
