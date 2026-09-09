"""Fixtures. Everything here is offline: no key, no network, no bill.

The suite runs against ``FakeClient`` — the third implementation of the model
boundary, and the one that makes the tests fast and deterministic. A handful of
tests marked ``@pytest.mark.gateway`` need the course gateway running; they skip
cleanly when it is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "infra" / "mockgw"))

from retail_support.domain.session import Session  # noqa: E402
from retail_support.llm.fake import FakeClient  # noqa: E402
from retail_support.tools.services import return_service, escalation_service  # noqa: E402


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def session() -> Session:
    return_service.reset()
    escalation_service.reset()
    return Session(customer_id="customer-A")


@pytest.fixture
def session_b() -> Session:
    return Session(customer_id="customer-B")


@pytest.fixture(autouse=True)
def _clean_services():
    return_service.reset()
    escalation_service.reset()
    yield
    return_service.reset()
    escalation_service.reset()


def gateway_url() -> str:
    """Follow the configured route rather than assuming a port.

    The suite has to work when the gateway is on 8080 (the documented default),
    on another port during development, or inside a compose network in CI — and
    the route configuration already knows which. Asking it is one line; guessing
    is a skipped test that looks like a passing one.
    """
    import os

    base = os.environ.get("RETAIL_SUPPORT_PRIMARY_BASE_URL", "http://127.0.0.1:8080/v1")
    return base.rstrip("/").removesuffix("/v1") + "/healthz"


def gateway_up(url: str | None = None) -> bool:
    import urllib.request

    url = url or gateway_url()
    try:
        with urllib.request.urlopen(url, timeout=1) as response:  # noqa: S310
            return response.status == 200
    except Exception:
        return False


@pytest.fixture
def requires_gateway():
    if not gateway_up():
        pytest.skip("course gateway not running (make gateway)")
