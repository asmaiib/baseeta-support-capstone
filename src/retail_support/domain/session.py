"""Conversation state, the PII vault, and the authorisation gate.

The load-bearing idea, and the one the SIM-swap case study exists to make
unforgettable: **authority lives here, not in the token stream.** Chat history is
data. Tool arguments are user input by proxy. Whether a booking may happen is
decided by ``Session.authorize`` against the *authenticated* customer — never by
anything a model said, however convincingly it said it.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from retail_support.llm.interfaces import Message
from retail_support.observability import get_logger

log = get_logger(__name__)

#: Saudi-specific patterns, used inbound (mask before any model or log sees them)
#: and outbound (nothing unmasked leaves the system).
SAUDI_PII: dict[str, re.Pattern[str]] = {
    "national_id": re.compile(r"\b[12]\d{9}\b"),
    "phone": re.compile(r"(?:\+?966|0)5\d{8}\b"),
    "iban": re.compile(r"\bSA\d{22}\b"),
}


class PIIVault:
    """Session-scoped, reversible masking.

    The mask round-trips *inside* the trust boundary only: the booking flow needs
    the real value at the tool gate, and nothing outside this process ever does.
    """

    def __init__(self) -> None:
        self._by_token: dict[str, str] = {}
        self._by_value: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def store(self, kind: str, value: str) -> str:
        if value in self._by_value:
            return self._by_value[value]
        self._counters[kind] = self._counters.get(kind, 0) + 1
        token = f"⟦{kind.upper()}_{self._counters[kind]}⟧"
        self._by_token[token] = value
        self._by_value[value] = token
        return token

    def reveal(self, token: str) -> str | None:
        return self._by_token.get(token)

    def unmask(self, text: str) -> str:
        for token, value in self._by_token.items():
            text = text.replace(token, value)
        return text

    def __len__(self) -> int:
        return len(self._by_token)


class AuthorizationVerdict(BaseModel):
    allowed: bool
    reason: str = ""
    user_hint: str = ""


@dataclass
class ConversationState:
    """Windowed history: last ``max_turns`` exchanges plus the system prompt.

    Full history is simple and grows unboundedly (cost linear in turn number,
    then a context-length crash for your most engaged users first). Summarised
    history is the capstone extension. Windowed is what RetailSupport ships.
    """

    max_turns: int = 8
    turns: list[Message] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.turns.append(Message(role="user", content=text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.turns.append(Message(role="assistant", content=text))
        self._trim()

    def _trim(self) -> None:
        # One "turn" is a user message plus its assistant reply.
        limit = self.max_turns * 2
        if len(self.turns) > limit:
            self.turns = self.turns[-limit:]

    def messages(self, system: str | None = None) -> list[Message]:
        head = [Message(role="system", content=system)] if system else []
        return head + list(self.turns)

    def clear(self) -> None:
        self.turns.clear()


class Session:
    """Everything the application knows that the conversation may not assert."""

    def __init__(
        self,
        customer_id: str = "customer-A",
        *,
        identity_verified: bool = True,
        max_turns: int = 8,
        language: str = "en",
    ) -> None:
        self.id = f"sess_{uuid.uuid4().hex[:10]}"
        self.customer_id = customer_id
        self.language = language
        self.pii_vault = PIIVault()
        self.state = ConversationState(max_turns=max_turns)
        self.identity_verified_at: datetime | None = (
            datetime.now(UTC) if identity_verified else None
        )
        self.verification_ttl = timedelta(minutes=15)
        #: Idempotency: a retried turn must not book twice.
        self.completed_side_effects: dict[str, dict] = {}
        self.tool_trace: list[dict] = []

    # --- identity --------------------------------------------------------
    @property
    def identity_fresh(self) -> bool:
        if self.identity_verified_at is None:
            return False
        return datetime.now(UTC) - self.identity_verified_at <= self.verification_ttl

    def verify_identity(self) -> None:
        """Called by the *application* after an out-of-band check completes.

        Never called because a conversation claimed verification had happened.
        """
        self.identity_verified_at = datetime.now(UTC)

    # --- authorisation ---------------------------------------------------
    def idempotency_key(self, tool_name: str, args: dict) -> str:
        parts = [tool_name] + [f"{k}={args[k]}" for k in sorted(args)]
        return "|".join(parts)

    def authorize(self, tool_name: str, args: dict) -> AuthorizationVerdict:
        """The gate every side-effecting tool passes through."""
        if not self.identity_fresh:
            return AuthorizationVerdict(
                allowed=False,
                reason="identity_not_verified",
                user_hint=(
                    "I need to verify your identity before I can do that. "
                    "Please complete the verification step and try again."
                ),
            )
        on_behalf_of = args.get("on_behalf_of") or args.get("customer_id")
        if on_behalf_of and on_behalf_of != self.customer_id:
            # The model's argument is user input by proxy. The session decides.
            log.warning(
                "authz_cross_customer_denied",
                session=self.id,
                tool=tool_name,
                requested_for=on_behalf_of,
            )
            return AuthorizationVerdict(
                allowed=False,
                reason="cross_customer",
                user_hint=(
                    "I can only act on your own account. Each person books their "
                    "own appointment from their own account."
                ),
            )
        key = self.idempotency_key(tool_name, args)
        if key in self.completed_side_effects:
            return AuthorizationVerdict(
                allowed=False,
                reason="already_done",
                user_hint="That is already booked — I won't book it twice.",
            )
        return AuthorizationVerdict(allowed=True)

    def record_side_effect(self, tool_name: str, args: dict, result: dict) -> None:
        self.completed_side_effects[self.idempotency_key(tool_name, args)] = result

    def replay_side_effect(self, tool_name: str, args: dict) -> dict | None:
        return self.completed_side_effects.get(self.idempotency_key(tool_name, args))


def mask_pii(text: str, session: Session) -> str:
    """Mask BEFORE any model or log sees the text."""
    for kind, pattern in SAUDI_PII.items():
        for match in list(pattern.finditer(text)):
            token = session.pii_vault.store(kind, match.group())
            text = text.replace(match.group(), token)
    return text


def contains_pii(text: str) -> str | None:
    for kind, pattern in SAUDI_PII.items():
        if pattern.search(text):
            return kind
    return None
