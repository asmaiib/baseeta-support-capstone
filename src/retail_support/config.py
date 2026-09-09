"""Configuration: model names are configuration, not code (Module 1 §3).

``retail_support-default`` resolves through this layer to a concrete model id. Swapping a
provider, pinning a snapshot, or moving a route on-premise is a YAML edit and an
env var — never a code change. That claim is tested in ``tests/llm/test_config.py``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retail_support.yaml"

Dialect = Literal["openai", "anthropic", "fake"]


class ModelRoute(BaseModel):
    """One way of reaching a model. Three deployments, one adapter (Module 2 §1)."""

    name: str
    dialect: Dialect = "openai"
    base_url: str | None = None
    api_key: SecretStr = SecretStr("")
    aliases: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 30.0
    connect_timeout_s: float = 5.0
    #: Marks a route as residency-clean or not. Routing policy, not a code branch.
    residency: Literal["cloud", "on_premise"] = "cloud"

    def resolve(self, alias: str) -> str:
        """Alias -> concrete model id. Unknown aliases are a *config* bug, loudly."""
        try:
            return self.aliases[alias]
        except KeyError as exc:
            raise KeyError(
                f"route {self.name!r} has no alias {alias!r}; "
                f"known aliases: {sorted(self.aliases)}"
            ) from exc


class PriceRow(BaseModel):
    """Illustrative course rates, in SAR per 1M tokens. Refresh every delivery."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float = 0.0


class PriceSheet(BaseModel):
    models: dict[str, PriceRow] = Field(default_factory=dict)
    default: PriceRow = PriceRow(input_per_mtok=0.0, output_per_mtok=0.0)

    def for_model(self, model_id: str) -> PriceRow:
        if model_id in self.models:
            return self.models[model_id]
        for key, row in self.models.items():  # prefix match tolerates dated snapshots
            if model_id.startswith(key):
                return row
        return self.default


class GuardSettings(BaseModel):
    max_input_chars: int = 4000
    classifier_alias: str = "retail_support-guard"
    classifier_enabled: bool = True
    canary: str = "⟦BASEETA-9c2e⟧"


class CacheSettings(BaseModel):
    enabled: bool = False
    backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    semantic_enabled: bool = False
    semantic_threshold: float = 0.90
    #: Arabic morphology puts near-misses closer together; Module 6 measures this.
    semantic_threshold_by_language: dict[str, float] = Field(default_factory=dict)
    ttl_seconds: int = 3600


class PipelineSettings(BaseModel):
    max_history_turns: int = 8
    max_tool_iterations: int = 6
    routing_table: dict[str, str | None] = Field(
        default_factory=lambda: {
            "faq": "retail_support-small",
            "service": "retail_support-default",
            "complex": "retail_support-flagship",
            "escalate": None,
        }
    )
    routing_enabled: bool = False  # Module 6 turns this on, eval-gated
    #: Cheap-first, escalate on a deterministic failure signal. The router pays
    #: one classification everywhere; the cascade pays double only on escalation.
    #: Traffic shape picks the winner, and the meter supplies the traffic shape.
    cascade_enabled: bool = False
    cascade_escalate_alias: str = "retail_support-flagship"


class Settings(BaseModel):
    routes: dict[str, ModelRoute]
    prices: PriceSheet = Field(default_factory=PriceSheet)
    guards: GuardSettings = Field(default_factory=GuardSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    primary_route: str = "primary"
    fallback_route: str | None = "vllm"
    cheap_route: str = "cheap"

    def route(self, name: str) -> ModelRoute:
        try:
            return self.routes[name]
        except KeyError as exc:
            raise KeyError(
                f"no route {name!r} configured; known routes: {sorted(self.routes)}"
            ) from exc


def _env_override(route_name: str, field: str) -> str | None:
    """``RETAIL_SUPPORT_PRIMARY_BASE_URL`` / ``RETAIL_SUPPORT_PRIMARY_API_KEY`` and friends.

    Keys come from the environment, never from the YAML committed to git.
    """
    return os.environ.get(f"RETAIL_SUPPORT_{route_name.upper()}_{field.upper()}")


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.environ.get("RETAIL_SUPPORT_CONFIG") or DEFAULT_CONFIG)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    routes: dict[str, ModelRoute] = {}
    for name, spec in (raw.get("routes") or {}).items():
        spec = dict(spec)
        spec["name"] = name
        if (base_url := _env_override(name, "base_url")) is not None:
            spec["base_url"] = base_url
        if (api_key := _env_override(name, "api_key")) is not None:
            spec["api_key"] = api_key
        routes[name] = ModelRoute(**spec)

    settings = Settings(
        routes=routes,
        prices=PriceSheet(**(raw.get("prices") or {})),
        guards=GuardSettings(**(raw.get("guards") or {})),
        cache=CacheSettings(**(raw.get("cache") or {})),
        pipeline=PipelineSettings(**(raw.get("pipeline") or {})),
        primary_route=raw.get("primary_route", "primary"),
        fallback_route=raw.get("fallback_route", "vllm"),
        cheap_route=raw.get("cheap_route", "cheap"),
    )
    if os.environ.get("RETAIL_SUPPORT_CACHE_ENABLED"):
        settings.cache.enabled = os.environ["RETAIL_SUPPORT_CACHE_ENABLED"] == "1"
    if os.environ.get("RETAIL_SUPPORT_SEMANTIC_CACHE_ENABLED"):
        settings.cache.semantic_enabled = os.environ["RETAIL_SUPPORT_SEMANTIC_CACHE_ENABLED"] == "1"
    if os.environ.get("RETAIL_SUPPORT_ROUTING_ENABLED"):
        settings.pipeline.routing_enabled = os.environ["RETAIL_SUPPORT_ROUTING_ENABLED"] == "1"
    if os.environ.get("RETAIL_SUPPORT_CASCADE_ENABLED"):
        settings.pipeline.cascade_enabled = os.environ["RETAIL_SUPPORT_CASCADE_ENABLED"] == "1"
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
