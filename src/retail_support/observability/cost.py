"""The cost meter: ``Usage`` finally monetised (Module 6 §1).

Rates live in config because they change quarterly. Costs are computed, logged and
aggregated by route and intent, so "where does the money actually go?" is a query
and not a guess. Metering comes *before* optimising — teams who invert that order
spend a week saving 4% while the 60% line item sits unexamined.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from retail_support.config import PriceSheet
from retail_support.llm.interfaces import LLMResponse
from retail_support.observability import current_trace_id, get_logger

log = get_logger(__name__)


class CostRecord(BaseModel):
    route: str
    intent: str
    stage: str = ""
    model_id: str
    prompt_version: str = ""
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    latency_ms: float = 0.0
    cost_halalas: float  # SAR cents. Money is never a float in production; fine for a meter.
    trace_id: str = ""
    cache_tier: str = ""  # "", "exact", "semantic" — a served-from-cache call costs nothing


class CostMeter:
    """Collects records in memory and (optionally) appends them as JSON lines."""

    def __init__(self, prices: PriceSheet, sink: str | Path | None = None) -> None:
        self._prices = prices
        self.records: list[CostRecord] = []
        self._sink = Path(sink) if sink else None
        if self._sink:
            self._sink.parent.mkdir(parents=True, exist_ok=True)

    def price_of(self, model_id: str, usage) -> float:
        p = self._prices.for_model(model_id)
        fresh_in = max(usage.input_tokens - usage.cached_input_tokens, 0)
        return (
            fresh_in * p.input_per_mtok
            + usage.cached_input_tokens * p.cached_input_per_mtok
            + usage.output_tokens * p.output_per_mtok
        ) / 1_000_000

    def meter(
        self,
        response: LLMResponse,
        *,
        route: str,
        intent: str = "unknown",
        stage: str = "",
        prompt_version: str = "",
        cache_tier: str = "",
    ) -> CostRecord:
        cost_sar = 0.0 if cache_tier else self.price_of(response.model_id, response.usage)
        record = CostRecord(
            route=route,
            intent=intent,
            stage=stage,
            model_id=response.model_id,
            prompt_version=prompt_version,
            input_tokens=response.usage.input_tokens,
            cached_tokens=response.usage.cached_input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=round(response.latency_ms, 1),
            cost_halalas=round(cost_sar * 100, 6),
            trace_id=current_trace_id(),
            cache_tier=cache_tier,
        )
        self.records.append(record)
        log.info("llm_cost", **record.model_dump())
        if self._sink:
            with self._sink.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
        return record

    # --- aggregation -----------------------------------------------------
    @property
    def total_halalas(self) -> float:
        return round(sum(r.cost_halalas for r in self.records), 4)

    def by(self, field: str) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for r in self.records:
            out[getattr(r, field)] += r.cost_halalas
        return {k: round(v, 4) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}

    def cached_input_share(self) -> float:
        total_in = sum(r.input_tokens for r in self.records)
        cached = sum(r.cached_tokens for r in self.records)
        return (cached / total_in) if total_in else 0.0

    def reset(self) -> None:
        self.records.clear()
