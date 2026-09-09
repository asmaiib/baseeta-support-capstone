"""Two-tier response cache. The exact tier is safe and boring. The semantic tier
is a quality trade, and it is treated with the suspicion a quality trade deserves.

Keys carry everything that could change the answer — model, prompt version, the
rendered prompt, the sampling parameters. A key that omits the prompt version
serves yesterday's behaviour after today's deploy.

The semantic tier is scoped by language *and* intent, and **only impersonal
content is ever eligible**. Anything conditioned on session state is excluded by
construction rather than by a condition someone might later edit — that is the
difference between a cost optimisation and a data-protection incident.

Two numbers, always reported together: hit rate **and** wrong-hit rate. The
pairing rule from the guards has an exact analogue here. A 40% hit rate with one
wrong hit per thousand is not a saving for a government assistant.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Protocol

from retail_support.caching.embeddings import cosine, embed
from retail_support.observability import get_logger

log = get_logger(__name__)


@dataclass
class CacheScope:
    """What makes two questions the same question."""

    language: str = "en"
    intent: str = "faq"
    #: The whole safety argument in one flag. Personalised content is never
    #: semantically cacheable — not "usually not", never.
    personalised: bool = True

    @property
    def semantic_eligible(self) -> bool:
        return not self.personalised and self.intent == "faq"

    @property
    def name(self) -> str:
        return f"{self.intent}:{self.language}:{'personal' if self.personalised else 'impersonal'}"


class KeyValueBackend(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_seconds: int) -> None: ...
    def clear(self) -> None: ...


class MemoryBackend:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            del self._data[key]
            return None
        return value

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._data[key] = (time.time() + ttl_seconds, value)

    def clear(self) -> None:
        self._data.clear()


class RedisBackend:
    """The compose stack's Redis. Exact tier only — see the note in the class docs."""

    def __init__(self, url: str) -> None:
        import redis  # imported here so Redis stays optional

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._client.setex(key, ttl_seconds, value)

    def clear(self) -> None:
        self._client.flushdb()


@dataclass
class SemanticEntry:
    vector: tuple[float, ...]
    query: str
    payload: str
    stored_at: float


@dataclass
class CacheStats:
    lookups: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    stores: int = 0
    #: Populated by ``make eval-cache``: a hit that returned the wrong answer.
    wrong_hits: int = 0
    near_miss_checks: int = 0
    #: Best neighbour score per lookup that did NOT hit. Without this you cannot
    #: answer the only question that matters when tuning a threshold: how much
    #: traffic is sitting just below it, and what would letting it through cost?
    near_scores: list[float] = dataclass_field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return ((self.exact_hits + self.semantic_hits) / self.lookups) if self.lookups else 0.0

    @property
    def wrong_hit_rate(self) -> float:
        return (self.wrong_hits / self.near_miss_checks) if self.near_miss_checks else 0.0

    @property
    def closest_non_hits(self) -> list[float]:
        return sorted(self.near_scores, reverse=True)[:5]

    def render(self) -> str:
        line = (
            f"lookups {self.lookups} | exact {self.exact_hits} | semantic {self.semantic_hits} "
            f"| hit rate {self.hit_rate:.0%} | wrong hits {self.wrong_hits}/{self.near_miss_checks}"
        )
        if self.closest_non_hits:
            line += f" | closest non-hits {self.closest_non_hits}"
        return line


class ResponseCache:
    def __init__(
        self,
        backend: KeyValueBackend | None = None,
        *,
        threshold: float = 0.90,
        threshold_by_language: dict[str, float] | None = None,
        ttl_seconds: int = 3600,
        semantic_enabled: bool = False,
    ) -> None:
        self._backend = backend or MemoryBackend()
        self._threshold = threshold
        self._threshold_by_language = threshold_by_language or {}
        self._ttl = ttl_seconds
        self.semantic_enabled = semantic_enabled
        self._semantic: dict[str, list[SemanticEntry]] = {}
        self.stats = CacheStats()

    # --- keys ------------------------------------------------------------
    @staticmethod
    def exact_key(model_id: str, prompt_version: str, rendered: str, params: dict) -> str:
        blob = json.dumps(
            [model_id, prompt_version, rendered, params], sort_keys=True, ensure_ascii=False
        )
        return "exact:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def threshold_for(self, language: str) -> float:
        return self._threshold_by_language.get(language, self._threshold)

    # --- lookup ----------------------------------------------------------
    def get(self, *, scope: CacheScope, query: str, exact_key: str) -> tuple[Any, str] | None:
        self.stats.lookups += 1
        raw = self._backend.get(exact_key)
        if raw is not None:
            self.stats.exact_hits += 1
            log.info("cache_hit", tier="exact", scope=scope.name)
            return json.loads(raw), "exact"

        if not (self.semantic_enabled and scope.semantic_eligible):
            return None

        index = f"sem:{scope.language}:{scope.intent}"
        vector = embed(query)
        best: SemanticEntry | None = None
        best_score = 0.0
        now = time.time()
        for entry in self._semantic.get(index, []):
            if now - entry.stored_at > self._ttl:
                continue
            score = cosine(vector, entry.vector)
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= self.threshold_for(scope.language):
            self.stats.semantic_hits += 1
            log.info(
                "cache_hit", tier="semantic", scope=scope.name, score=round(best_score, 3)
            )
            return json.loads(best.payload), "semantic"
        if best is not None:
            self.stats.near_scores.append(round(best_score, 3))
        return None

    def put(self, *, scope: CacheScope, query: str, exact_key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        self._backend.set(exact_key, payload, self._ttl)
        self.stats.stores += 1
        if self.semantic_enabled and scope.semantic_eligible:
            index = f"sem:{scope.language}:{scope.intent}"
            self._semantic.setdefault(index, []).append(
                SemanticEntry(embed(query), query, payload, time.time())
            )

    def clear(self) -> None:
        self._backend.clear()
        self._semantic.clear()
        self.stats = CacheStats()


def build_cache(settings) -> ResponseCache:
    backend: KeyValueBackend
    if settings.cache.backend == "redis":
        try:
            backend = RedisBackend(settings.cache.redis_url)
            backend.get("warm")
        except Exception as exc:  # noqa: BLE001
            log.warning("redis_unavailable", error=type(exc).__name__, fallback="memory")
            backend = MemoryBackend()
    else:
        backend = MemoryBackend()
    return ResponseCache(
        backend,
        threshold=settings.cache.semantic_threshold,
        threshold_by_language=settings.cache.semantic_threshold_by_language,
        ttl_seconds=settings.cache.ttl_seconds,
        semantic_enabled=settings.cache.semantic_enabled,
    )
