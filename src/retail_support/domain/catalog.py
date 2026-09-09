"""The product & policy catalog: the only trusted source of retail facts.

Rendered into the FAQ prompt's *stable prefix* (Module 6 §3 — byte-stable, so
the provider caches it), delimited as trusted content, and used by the
groundedness check as the ground truth an answer is graded against.

Rendering is deliberately deterministic: the same catalog renders
byte-identical every time. A rendering that sorted a dict or stamped a date
would quietly destroy the prompt cache.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "product_catalog.yaml"

Language = Literal["ar", "en"]


class Bilingual(BaseModel):
    en: str = ""
    ar: str = ""

    def get(self, language: str) -> str:
        return self.ar if language == "ar" else self.en


class BilingualList(BaseModel):
    en: list[str] = Field(default_factory=list)
    ar: list[str] = Field(default_factory=list)

    def get(self, language: str) -> list[str]:
        return self.ar if language == "ar" else self.en


class CategoryEntry(BaseModel):
    id: str
    category: str
    title: Bilingual
    return_window: Bilingual
    warranty: Bilingual
    shipping_fee: Bilingual
    restocking_fee: Bilingual
    keywords: BilingualList

    def all_keywords(self) -> list[str]:
        return [k.lower() for k in self.keywords.en] + list(self.keywords.ar)


class ProductCatalog(BaseModel):
    version: str
    store_name: Bilingual
    entries: list[CategoryEntry]
    general_policy: dict[str, Bilingual]
    store_locations: BilingualList

    def by_id(self, entry_id: str) -> CategoryEntry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def by_category(self, category: str) -> CategoryEntry | None:
        return next((e for e in self.entries if e.category == category), None)

    def render(self, language: Language = "en") -> str:
        """Markdown, stable byte-for-byte for a given catalog version+language."""
        lines = [f"Catalog version: {self.version}", f"store: {self.store_name.get(language)}", ""]
        for e in self.entries:
            lines.append(f"### {e.id} — {e.title.get(language)}")
            lines.append(f"- category: {e.category}")
            lines.append(f"- return_window: {e.return_window.get(language)}")
            lines.append(f"- warranty: {e.warranty.get(language)}")
            lines.append(f"- shipping_fee: {e.shipping_fee.get(language)}")
            lines.append(f"- restocking_fee: {e.restocking_fee.get(language)}")
            lines.append("- keywords: " + ", ".join(e.keywords.get(language)))
            lines.append("")
        lines.append("### general_policy")
        for key, value in self.general_policy.items():
            lines.append(f"- {key}: {value.get(language)}")
        lines.append("")
        lines.append("store_locations: " + ", ".join(self.store_locations.get(language)))
        return "\n".join(lines)


@lru_cache(maxsize=4)
def load_catalog(path: str | Path | None = None) -> ProductCatalog:
    raw = yaml.safe_load(Path(path or CATALOG_PATH).read_text(encoding="utf-8"))
    return ProductCatalog(**raw)


@lru_cache(maxsize=8)
def rendered_catalog(language: Language = "en") -> str:
    return load_catalog().render(language)
