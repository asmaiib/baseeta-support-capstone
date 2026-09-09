"""Configuration, caching and the meter — the three things that make a provider
swap a YAML edit, a saving safe, and a bill explicable.
"""

from __future__ import annotations

import json

import pytest

from retail_support.caching.embeddings import similarity
from retail_support.caching.response_cache import CacheScope, MemoryBackend, ResponseCache
from retail_support.config import PriceRow, PriceSheet, load_settings
from retail_support.llm.interfaces import LLMResponse, Usage
from retail_support.observability.cost import CostMeter

# --- configuration --------------------------------------------------------


def test_aliases_resolve_to_concrete_models_per_route():
    settings = load_settings()
    assert settings.route("primary").resolve("retail_support-default") == "course-flagship"
    assert settings.route("cheap").resolve("retail_support-default") == "course-small"
    assert settings.route("vllm").resolve("retail_support-default") == "retail_support-onprem"


def test_an_unknown_alias_is_a_loud_config_error():
    settings = load_settings()
    with pytest.raises(KeyError, match="no alias"):
        settings.route("primary").resolve("retail_support-nonexistent")


def test_env_overrides_the_base_url_without_touching_code(monkeypatch):
    monkeypatch.setenv("RETAIL_SUPPORT_VLLM_BASE_URL", "http://gpu-server.classroom.local:8000/v1")
    settings = load_settings()
    assert settings.route("vllm").base_url == "http://gpu-server.classroom.local:8000/v1"


def test_residency_is_configuration_not_a_code_branch():
    settings = load_settings()
    assert settings.route("vllm").residency == "on_premise"
    assert settings.route("primary").residency == "cloud"


def test_no_api_key_is_committed_in_the_config():
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "configs" / "retail_support.yaml").read_text(
        encoding="utf-8"
    )
    for marker in ("sk-", "sk-ant-", "AKIA"):
        assert marker not in text, f"a real-looking key ({marker}) is committed in the config"


# --- the meter ------------------------------------------------------------


def prices() -> PriceSheet:
    return PriceSheet(
        models={
            "flagship": PriceRow(
                input_per_mtok=10.0, output_per_mtok=50.0, cached_input_per_mtok=1.0
            )
        }
    )


def test_cached_tokens_are_billed_at_the_cached_rate():
    meter = CostMeter(prices())
    response = LLMResponse(
        model_id="flagship",
        usage=Usage(input_tokens=1000, output_tokens=100, cached_input_tokens=800),
    )
    record = meter.meter(response, route="primary", intent="faq")
    # 200 fresh at 10 + 800 cached at 1 + 100 output at 50 = 2000 + 800 + 5000 per Mtok
    assert record.cost_halalas == pytest.approx((200 * 10 + 800 * 1 + 100 * 50) / 1e6 * 100)


def test_a_cache_served_call_costs_nothing():
    meter = CostMeter(prices())
    response = LLMResponse(model_id="flagship", usage=Usage(input_tokens=1000, output_tokens=100))
    record = meter.meter(response, route="primary", intent="faq", cache_tier="exact")
    assert record.cost_halalas == 0.0


def test_the_meter_aggregates_by_intent_and_stage():
    meter = CostMeter(prices())
    for intent in ("faq", "faq", "service"):
        meter.meter(
            LLMResponse(model_id="flagship", usage=Usage(input_tokens=100, output_tokens=10)),
            route="primary",
            intent=intent,
            stage=f"{intent}_handler",
        )
    by_intent = meter.by("intent")
    assert set(by_intent) == {"faq", "service"}
    assert by_intent["faq"] > by_intent["service"]


def test_the_meter_writes_jq_able_json_lines(tmp_path):
    sink = tmp_path / "llm_cost.jsonl"
    meter = CostMeter(prices(), sink=sink)
    meter.meter(
        LLMResponse(model_id="flagship", usage=Usage(input_tokens=10, output_tokens=1)),
        route="primary",
        intent="faq",
    )
    rows = [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["intent"] == "faq"
    assert "cost_halalas" in rows[0]


# --- the cache ------------------------------------------------------------


def test_the_exact_key_covers_everything_that_changes_an_answer():
    base = ResponseCache.exact_key("m", "answer_faq.v5", "rendered", {"temperature": 0.4})
    assert base != ResponseCache.exact_key("m", "answer_faq.v6", "rendered", {"temperature": 0.4})
    assert base != ResponseCache.exact_key("m2", "answer_faq.v5", "rendered", {"temperature": 0.4})
    assert base != ResponseCache.exact_key("m", "answer_faq.v5", "other", {"temperature": 0.4})
    assert base != ResponseCache.exact_key("m", "answer_faq.v5", "rendered", {"temperature": 0.7})


def test_personalised_content_is_never_semantically_cacheable():
    assert not CacheScope(personalised=True).semantic_eligible
    assert not CacheScope(personalised=False, intent="service").semantic_eligible
    assert CacheScope(personalised=False, intent="faq").semantic_eligible


def test_a_personalised_scope_gets_no_semantic_hit_however_similar():
    cache = ResponseCache(MemoryBackend(), threshold=0.5, semantic_enabled=True)
    impersonal = CacheScope(personalised=False)
    personal = CacheScope(personalised=True)
    cache.put(scope=impersonal, query="how do I return my order", exact_key="k1", value={"t": 1})
    assert cache.get(scope=personal, query="how do I return my order!", exact_key="k2") is None


def test_the_semantic_tier_serves_a_paraphrase_and_refuses_a_near_miss():
    cache = ResponseCache(MemoryBackend(), threshold=0.86, semantic_enabled=True)
    scope = CacheScope(personalised=False)
    cache.put(
        scope=scope,
        query="What is your return policy for electronics?",
        exact_key="k1",
        value={"text": "return policy answer"},
    )
    paraphrase = cache.get(
        scope=scope, query="What is your return policy on electronics?", exact_key="k2"
    )
    near_miss = cache.get(
        scope=scope, query="What is your cancellation policy for electronics?", exact_key="k3"
    )
    assert paraphrase is not None and paraphrase[1] == "semantic"
    assert near_miss is None, "return and cancellation are not the same question"


def test_near_miss_scores_are_recorded_for_threshold_tuning():
    cache = ResponseCache(MemoryBackend(), threshold=0.99, semantic_enabled=True)
    scope = CacheScope(personalised=False)
    cache.put(scope=scope, query="What is your return policy?", exact_key="k1", value={"t": 1})
    cache.get(scope=scope, query="What's your return policy?", exact_key="k2")
    assert cache.stats.closest_non_hits, (
        "you cannot tune a threshold without knowing what is sitting just below it"
    )


def test_arabic_needs_a_higher_threshold_than_english():
    """Measured, not assumed: Arabic spelling variants of the *same* question
    score higher than English paraphrases, so Arabic sits closer to its
    near-misses and needs more margin, not less."""
    ar_variant = similarity("ما هي سياسة الإرجاع للإلكترونيات؟", "ما هي سياسة الارجاع للالكترونيات")
    en_variant = similarity("What is your return policy for electronics?", "What's your return policy on electronics?")
    assert ar_variant > en_variant
