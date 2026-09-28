"""Read a job's HF-reported inference cost. Missing usage is pending, never $0."""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Optional
from urllib.parse import quote

import requests


def fetch_session_cost(
    *,
    api_key: str,
    bill_to: Optional[str],
    session_id: str,
    created_at: float,
    now: float,
) -> dict[str, Any]:
    namespace = f"organizations/{quote(bill_to, safe='')}" if bill_to else "settings"
    # Sessions can cross a month boundary; request all affected periods.
    start = datetime.fromtimestamp(created_at, timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    result: dict[str, Any] = {
        "status": "pending",
        "session_id": session_id,
        "currency": "USD",
        "cost_usd": None,
        "request_count": None,
        "checked_at": now,
    }
    try:
        response = requests.get(
            f"https://huggingface.co/api/{namespace}/billing/usage-by-inference-session",
            headers={"Authorization": f"Bearer {api_key}"},
            params={
                "startDate": start.isoformat().replace("+00:00", "Z"),
                "endDate": datetime.fromtimestamp(now, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            },
            timeout=10,
            allow_redirects=False,
        )
        if response.status_code in (401, 403):
            return {**result, "status": "forbidden"}
        if response.status_code != 200:
            return {**result, "status": "unavailable"}
        payload = response.json()
        if payload["currency"] != "USD":
            raise ValueError("Unexpected billing currency")
        cost = Decimal(0)
        count = 0
        found = False
        for period in payload["periods"]:
            for session in period["sessions"]:
                if session["id"] != session_id:
                    continue
                amount = Decimal(str(session["costCents"]))
                requests_count = session["requestCount"]
                if (
                    not amount.is_finite()
                    or amount < 0
                    or type(requests_count) is not int
                    or requests_count < 0
                ):
                    raise ValueError("Invalid billing amount")
                cost += amount / 100
                count += requests_count
                found = True
        if found:
            result.update(status="reported", cost_usd=float(cost), request_count=count)
        return result
    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
        InvalidOperation,
    ):
        # Never log the response: it may carry other sessions' usage.
        return {**result, "status": "unavailable"}
