"""Module 2: the same twenty bilingual prompts against every configured route.

Output feeds ``BENCHMARKS.md``, then the Module 5 eval and the Module 6 cost model.
Report p50 *and* p95: the user experience is the tail's, and an average hides the
queueing, the retries and the cold caches that make the tail what it is.

    python scripts/bench_providers.py
    python scripts/bench_providers.py --routes primary,vllm --repeat 2
"""

from __future__ import annotations

import argparse
import json

from _common import bootstrap, read_jsonl, rule, summarise_latency, write_json

bootstrap()

from retail_support.app import build_client  # noqa: E402
from retail_support.config import get_settings  # noqa: E402
from retail_support.domain.catalog import rendered_catalog  # noqa: E402
from retail_support.llm.interfaces import LLMRequest, Message  # noqa: E402
from retail_support.observability.cost import CostMeter  # noqa: E402
from retail_support.prompts.registry import load_prompt  # noqa: E402


def bench(client, name: str, prompts: list[dict], meter: CostMeter, repeat: int) -> dict:
    prompt = load_prompt("answer_faq.v1")
    systems = {
        language: prompt.render(product_catalog=rendered_catalog(language))
        for language in ("en", "ar")
    }
    latencies: list[float] = []
    tokens_in = tokens_out = cached = 0
    cost = 0.0
    by_language: dict[str, list[float]] = {"en": [], "ar": []}
    token_by_language: dict[str, int] = {"en": 0, "ar": 0}

    for _ in range(repeat):
        for row in prompts:
            language = row["language"]
            response = client.complete(
                LLMRequest(
                    messages=[
                        Message(role="system", content=systems[language]),
                        Message(
                            role="user",
                            content=f"<customer_message>\n{row['user']}\n</customer_message>",
                        ),
                    ],
                    max_tokens=300,
                    temperature=0.2,
                    cache_prefix_messages=1,
                )
            )
            latencies.append(response.latency_ms)
            by_language[language].append(response.latency_ms)
            tokens_in += response.usage.input_tokens
            tokens_out += response.usage.output_tokens
            cached += response.usage.cached_input_tokens
            token_by_language[language] += response.usage.input_tokens
            cost += meter.price_of(response.model_id, response.usage)

    summary = summarise_latency(latencies)
    return {
        "route": name,
        **summary,
        "p50_ar_ms": summarise_latency(by_language["ar"])["p50_ms"],
        "p50_en_ms": summarise_latency(by_language["en"])["p50_ms"],
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cached_input_tokens": cached,
        "ar_over_en_input_tokens": round(
            token_by_language["ar"] / token_by_language["en"], 2
        )
        if token_by_language["en"]
        else 0.0,
        "sar_total": round(cost, 4),
        "halalas_per_call": round(cost * 100 / max(len(latencies), 1), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", default=None, help="comma-separated route names")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    settings = get_settings()
    names = args.routes.split(",") if args.routes else list(settings.routes)
    prompts = read_jsonl("bench_prompts.jsonl")
    meter = CostMeter(settings.prices)

    rule(f"bench-providers | {len(prompts)} bilingual prompts x {len(names)} routes")
    results = []
    for name in names:
        client = build_client(settings, name)
        row = bench(client, name, prompts, meter, args.repeat)
        row["residency"] = settings.route(name).residency
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    print()
    header = f"{'route':<12} {'p50':>7} {'p95':>7} {'p50 ar':>7} {'p50 en':>7} {'ar/en tok':>10} {'hal/call':>9}  residency"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['route']:<12} {row['p50_ms']:>6}ms {row['p95_ms']:>6}ms "
            f"{row['p50_ar_ms']:>6}ms {row['p50_en_ms']:>6}ms {row['ar_over_en_input_tokens']:>10} "
            f"{row['halalas_per_call']:>9}  {row['residency']}"
        )
    write_json("bench_providers.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
