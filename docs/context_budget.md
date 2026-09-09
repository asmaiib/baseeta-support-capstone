# The context budget

Written in Module 1. Not an afterthought: history plus system prompt plus retrieved
content plus tool schemas plus output must fit the window, and **cost scales with
all of it on every turn**. A budget written down is a design artefact; a budget
discovered in production is an incident.

## Baseeta's allocation, 16k-token request budget

| Line | Budget (tok) | Measured | Why this number |
|---|---|---|---|
| System prompt (`answer_faq.v1`) | 400 | ~110 chars of role text | Versioned and immutable, so it is byte-stable and the provider caches it. Growth here is a prompt PR, not a surprise. |
| Product catalog (trusted context) | 900 | 1,671 chars (en) / 1,520 chars (ar) | The whole catalog travels on every FAQ request. It is the cheapest place to be generous *because* it caches (measured 82.9% cached_input_share over the golden set); it would be the most expensive if it did not. |
| Tool schemas (service route only) | 900 | 3 tools, one per risk class | Every tool added is paid for on every turn of that route, forever. |
| Windowed history (8 turns) | 4,800 | ~600 tok/turn cap | Unbounded history costs linearly per turn and then overflows for your most engaged customers first. |
| This turn's customer message | 1,000 | ~40 typical, higher for a long complaint | The cap is enforced by the guard's `max_input_chars` (4,000 chars), not hoped for. |
| Output (`max_tokens`) | 700 | ~150 typical | Always bounded. `finish_reason: length` is a correctness bug, and it is logged as one. |
| **Total, FAQ route** | **~7,700** | well under budget in measured runs | Comfortable headroom against a 16k window. |

Character counts above are measured directly
(`retail_support.domain.catalog.rendered_catalog`); exact token counts need
`tiktoken`'s remote BPE file, which this sandboxed environment could not
reach (`403` fetching `o200k_base.tiktoken`) — re-run `make token-report` in
an environment with network access to get exact per-tokenizer figures, using
the ~4–5 chars/token (English) and ~3.3 chars/token (Arabic) ratios Module 2
measured as an approximation until then.

## What the budget forces

**Windowing, not summarisation, for this product.** At roughly 600 tokens/turn,
unbounded history would cross a 16k window within a few dozen turns, and it
costs linearly the whole way up. Windowing at 8 turns (`ConversationState`)
keeps the per-turn cost flat. Summarisation is the capstone extension: it buys
back long-conversation memory at the cost of one extra call and a new failure
mode — a summary that drops the fact the customer needed.

**The catalog is the budget's biggest fixed line, and its cheapest.**
Byte-stable and therefore cached: measured at 82.9% `cached_input_share` on
the golden-set run, which is why moving the "today is..." timestamp out of
the system prompt (the `answer_faq.v0` → `v1` fix) mattered — one dynamic
byte at the top of the prefix would have destroyed all of it (measured: 0%
cached with the timestamp in the prefix, 68% with it moved to the volatile
tail).

## The failure this prevents

Unbounded history: a crash at some turn count, in production, for the most
engaged customers. The forecast is arithmetic anybody can do in a minute —
which is exactly why it is worth doing before the customers do it for you.
