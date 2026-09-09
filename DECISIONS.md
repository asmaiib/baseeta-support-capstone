# Decisions

The architecture decision records live in [`docs/adr/`](docs/adr/). This page is
the index, plus the one section a review board always asks for: what did you
change your mind about?

| ADR | Decision | Status |
|---|---|---|
| [001](docs/adr/001-architecture-pattern.md) | Router-first, with exactly one bounded tool loop | accepted |
| [002](docs/adr/002-model-and-routing.md) | Model choice is a routing table, not a winner | accepted |
| [003](docs/adr/003-the-course-gateway.md) | Zero-key operation via a smart FakeClient default, not the HTTP gateway simulator | accepted |

## Trade-offs we reversed

### The router's "question mark means FAQ" heuristic, reversed once

**First decision.** Build `smart_default`'s intent router as a keyword list:
if the message contains a return/exchange/status word, route to `service`;
otherwise `faq`.

**What happened.** The first full golden-set run scored 86% (108/125). The
failures clustered in one place: FAQ questions containing the word "return"
("What is your return policy for electronics?") were being misrouted to
`service`, because "return" is also the strongest signal for an actual return
request. Growing the keyword list case by case (adding "policy openers" like
"what is", "how long", "is there" to claw back each failing case) briefly
worked but was visibly fragile — every new phrasing needed a new special
case, and Arabic needed its own parallel list.

**What we did not do.** Keep patching the opener list. It clears one failure
at a time and never converges; the golden set has more phrasings than any
list will anticipate.

**Second decision.** Reorder the router by elimination instead: check for an
order reference, an escalation word, or an imperative action phrase
("can you check…", "أبغى أرجع…") *first* — those are the only cases that are
genuinely `service`. Anything left that ends in a question mark is, by
elimination, informational.

**Result.** 125/125 (100%), every slice including `faq` and `ar`, with a
*smaller* rule set than the patched version, not a larger one. The lesson
generalises past this one function: a growing exception list is a signal the
rule is inverted, not that it needs one more exception.

### The domain simulator, built small on purpose

**First instinct.** Adapt the course's own `infra/mockgw` (an ~850-line HTTP
rule engine) to Baseeta's domain, since it is the reference implementation's
zero-key backend.

**What we did not do.** That 850-line engine is written for Murshid's
domain — government service types, `CR`/`TR` reference formats — and porting
it wholesale risked exactly the outcome the capstone brief warns against:
Murshid's shape with the labels swapped, rather than something built for this
traffic. It would also have been the single largest piece of this submission
by line count, for a component whose job is to be invisible.

**Second decision.** Give `FakeClient` — already generic, already used
throughout the test suite — an intelligent default responder instead
(`llm/fake_brain.py`, ~500 lines, retail-specific from the first line). See
[ADR 003](docs/adr/003-the-course-gateway.md) for the full reasoning and its
honestly-stated limits.

**Result.** Zero-key operation that is unambiguously Baseeta's own — every
keyword list, every regex, every routed intent was written against this
domain's traffic, not adapted from another one — at roughly half the size of
the component it replaces.
