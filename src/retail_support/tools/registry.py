"""Baseeta's three tools — one per risk class, deliberately.

Tool schemas are prompts wearing a type system. The model chooses tools by
reading names and descriptions, so both are written for the model and tested
like prompts:

* **names** are verb_noun and unambiguous.
* **descriptions** say when to use the tool, when *not* to, and what it
  returns.
* **parameters** prefer enums to free strings wherever the domain is closed,
  and document formats in the description.

The risk class is *in code*, reviewable at a glance, because the
authorisation gate keys off it and a gate that depends on someone
remembering is not a gate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from retail_support.tools.services import escalation_service, return_service, status_lookup

RiskClass = Literal["read_only", "side_effecting", "terminal"]


class Tool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict  # JSON Schema
    risk: RiskClass
    fn: Callable  # executed by the APPLICATION, never by the model

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


CITY_ENUM = ["Riyadh", "Jeddah", "Dammam", "Makkah", "Madinah", "Khobar"]
CATEGORY_ENUM = ["electronics", "home_goods", "fashion"]

TOOLS: list[Tool] = [
    Tool(
        name="check_order_status",
        risk="read_only",
        description=(
            "Look up the current status of an order by its reference number "
            "(format: ORD- followed by 6-10 digits, e.g. ORD-1000001). Use "
            "when the customer asks where an order is or what its status is. "
            "Do NOT use to start a return, and do NOT use when the customer "
            "has not given an order reference — ask for it instead. Returns "
            "the status, the date it was last updated, and any note."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_reference": {
                    "type": "string",
                    "pattern": "^ORD-[0-9]{6,10}$",
                    "description": "The order reference, e.g. ORD-1000001",
                }
            },
            "required": ["order_reference"],
            "additionalProperties": False,
        },
        fn=status_lookup,
    ),
    Tool(
        name="create_return_request",
        risk="side_effecting",
        description=(
            "File a return or exchange for the AUTHENTICATED customer's own "
            "order. Use only after the customer has explicitly confirmed the "
            "order reference, the resolution (refund or exchange), and a "
            "specific future working day to drop the item off. Do NOT use to "
            "check eligibility only, and do NOT use to file a return on "
            "behalf of anyone else — the portal will refuse it. Returns the "
            "case id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_reference": {
                    "type": "string",
                    "pattern": "^ORD-[0-9]{6,10}$",
                },
                "category": {"type": "string", "enum": CATEGORY_ENUM},
                "resolution": {"type": "string", "enum": ["refund", "exchange"]},
                "dropoff_city": {"type": "string", "enum": CITY_ENUM},
                "dropoff_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD. Must be a future working day (Sunday to Thursday).",
                },
            },
            "required": [
                "order_reference", "category", "resolution", "dropoff_city", "dropoff_date",
            ],
            "additionalProperties": False,
        },
        fn=return_service.create,
    ),
    Tool(
        name="escalate_to_agent",
        risk="terminal",
        description=(
            "Transfer the conversation to a human agent. Use when the "
            "customer asks for a person, is distressed, is making a "
            "complaint that needs a human decision, or the request is "
            "outside what self-service can do. This ends your part of the "
            "conversation — do not call anything after it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One short sentence for the agent picking this up.",
                }
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
        fn=escalation_service.handoff,
    ),
]

BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def tool_schemas(allowed: list[str] | None = None) -> list[dict]:
    """The allowed-tool list is per route. The FAQ handler gets *no* tools."""
    tools = TOOLS if allowed is None else [t for t in TOOLS if t.name in allowed]
    return [t.schema() for t in tools]
