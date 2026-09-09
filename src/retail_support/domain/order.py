"""``ReturnCase`` — the structured object Baseeta files into the returns system.

The division of labour Module 3 is about:

* the **JSON Schema** constrains *syntax* — the provider's strict mode
  guarantees the shape, the enums and that no extra keys appear;
* **pydantic validators** enforce *semantics* — that an order reference has
  the right shape, that a drop-off date is a real future day the stores are
  open. Constrained decoding cannot know either of those things.

Belt and suspenders, each doing a different job. Which means validation can
still fail, which means the application still needs a designed failure path.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CITIES = ("Riyadh", "Jeddah", "Dammam", "Makkah", "Madinah", "Khobar", "unknown")
CATEGORIES = ("electronics", "home_goods", "fashion", "other")
ISSUE_TYPES = ("return", "exchange", "order_status", "complaint", "other")

_ORDER_REF_RE = re.compile(r"^ORD-[0-9]{6,10}$")


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str = Field(description="Name exactly as given by the customer")
    phone: str | None = Field(None, description="Phone in +9665XXXXXXXX form, if provided")

    @field_validator("phone")
    @classmethod
    def valid_phone(cls, v: str | None) -> str | None:
        if v is None:
            return v
        digits = v.replace(" ", "").replace("-", "")
        if not (digits.startswith(("+9665", "05", "9665")) and sum(c.isdigit() for c in digits) >= 9):
            raise ValueError("must be a Saudi mobile number, e.g. +9665XXXXXXXX")
        return digits


class ReturnCase(BaseModel):
    """Extracted from a free-text customer message.

    Fields the message does not contain are ``None`` — the model must NOT
    invent them.
    """

    model_config = ConfigDict(extra="forbid")

    issue_type: Literal["return", "exchange", "order_status", "complaint", "other"]
    order_reference: str | None = Field(
        None, description="Order reference if the customer gave one, e.g. ORD-1234567"
    )
    category: Literal["electronics", "home_goods", "fashion", "other"]
    summary_en: str = Field(description="One-sentence English summary for the case system")
    city: Literal["Riyadh", "Jeddah", "Dammam", "Makkah", "Madinah", "Khobar", "unknown"]
    urgency: Literal["routine", "urgent", "emergency"]
    language: Literal["ar", "en", "mixed"]
    customer: Customer
    needs_human: bool = Field(description="True if the request cannot be served by self-service")

    @field_validator("order_reference")
    @classmethod
    def valid_order_reference(cls, v: str | None) -> str | None:
        if v is not None and not _ORDER_REF_RE.match(v):
            raise ValueError("order reference must look like ORD-1234567")
        return v


def strict_schema(model: type[BaseModel] = ReturnCase, name: str = "return_case") -> dict:
    """What goes over the wire, in the strict-mode subset.

    Strict mode is narrower than JSON Schema: objects must be closed
    (``additionalProperties: false``) and *every* property must be listed as
    required — optional fields are expressed as a nullable type, never by
    absence.
    """
    schema = model.model_json_schema()
    _tighten(schema, schema)
    schema["title"] = name
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def _tighten(node: dict, root: dict) -> None:
    if not isinstance(node, dict):
        return
    for sub in node.get("$defs", {}).values():
        _tighten(sub, root)
    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False
        node["required"] = sorted(node.get("properties", {}))
    for value in node.get("properties", {}).values():
        _tighten(value, root)
    for key in ("items", "anyOf", "allOf", "oneOf"):
        value = node.get(key)
        if isinstance(value, list):
            for item in value:
                _tighten(item, root)
        elif isinstance(value, dict):
            _tighten(value, root)


TICKET_SCHEMA = strict_schema()


def schema_violations(schema: dict) -> list[str]:
    """Human-readable reasons a schema would be rejected by strict mode."""
    problems: list[str] = []

    def walk(node, path="$"):
        if not isinstance(node, dict):
            return
        for name, sub in node.get("$defs", {}).items():
            walk(sub, f"$defs.{name}")
        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: object is open (needs additionalProperties: false)")
            declared = set(node.get("properties", {}))
            required = set(node.get("required", []))
            if declared - required:
                missing = ", ".join(sorted(declared - required))
                problems.append(f"{path}: properties not listed as required ({missing})")
        for key, sub in node.get("properties", {}).items():
            walk(sub, f"{path}.{key}")

    walk(schema.get("json_schema", {}).get("schema", schema))
    return problems


class ReturnRequest(BaseModel):
    """The side-effecting tool's argument contract. Validated before anything runs."""

    model_config = ConfigDict(extra="forbid")

    order_reference: str = Field(description="e.g. ORD-1234567")
    category: Literal["electronics", "home_goods", "fashion"]
    resolution: Literal["refund", "exchange"]
    dropoff_city: Literal["Riyadh", "Jeddah", "Dammam", "Makkah", "Madinah", "Khobar"]
    dropoff_date: dt.date = Field(description="YYYY-MM-DD, must be a future working day")

    @field_validator("order_reference")
    @classmethod
    def valid_ref(cls, v: str) -> str:
        if not _ORDER_REF_RE.match(v):
            raise ValueError("order reference must look like ORD-1234567")
        return v

    @field_validator("dropoff_date")
    @classmethod
    def future_working_day(cls, v: dt.date) -> dt.date:
        if v <= dt.date.today():
            raise ValueError("drop-off date must be in the future")
        if v > dt.date.today() + dt.timedelta(days=30):
            raise ValueError("drop-off appointments open 30 days ahead at most")
        if v.weekday() in (4, 5):  # Friday, Saturday
            raise ValueError("stores are closed on Friday and Saturday")
        return v
