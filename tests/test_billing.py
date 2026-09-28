from unittest.mock import Mock, patch

import pytest
import requests

from reviewbot.billing import fetch_session_cost


def fetch(payload=None, status=200):
    response = Mock(status_code=status)
    response.json.return_value = payload
    with patch("reviewbot.billing.requests.get", return_value=response) as get:
        result = fetch_session_cost(
            api_key="secret",
            bill_to="acme",
            session_id="task-1",
            created_at=1788220800,
            now=1790899200,
        )
    return result, get


def test_sums_only_this_task_across_months_and_preserves_fractional_cents():
    result, get = fetch(
        {
            "currency": "USD",
            "periods": [
                {
                    "sessions": [
                        {"id": "task-1", "costCents": 0.125, "requestCount": 2},
                        {"id": "other-task", "costCents": 9999, "requestCount": 90},
                    ]
                },
                {"sessions": [{"id": "task-1", "costCents": 12, "requestCount": 3}]},
            ],
        }
    )
    assert result["cost_usd"] == 0.12125
    assert result["request_count"] == 5
    assert result["status"] == "reported"
    assert (
        get.call_args.args[0]
        == "https://huggingface.co/api/organizations/acme/billing/usage-by-inference-session"
    )
    assert get.call_args.kwargs["params"] == {
        "startDate": "2026-09-01T00:00:00Z",
        "endDate": "2026-10-02T00:00:00Z",
    }
    assert "secret" not in str(result)


def test_missing_is_pending_but_reported_zero_is_zero():
    missing, _ = fetch({"currency": "USD", "periods": [{"sessions": []}]})
    assert missing["cost_usd"] is None
    assert missing["status"] == "pending"
    zero, _ = fetch(
        {
            "currency": "USD",
            "periods": [
                {"sessions": [{"id": "task-1", "costCents": 0, "requestCount": 1}]}
            ],
        }
    )
    assert zero["status"] == "reported"
    assert zero["cost_usd"] == 0


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "forbidden"),
        (403, "forbidden"),
        (429, "unavailable"),
        (500, "unavailable"),
        (302, "unavailable"),
    ],
)
def test_failed_reads_do_not_become_zero_cost(status, expected):
    result, _ = fetch(status=status)
    assert result["status"] == expected
    assert result["cost_usd"] is None


@pytest.mark.parametrize("amount", [-1, "NaN", "Infinity", None, "invalid"])
def test_invalid_amounts_are_unavailable(amount):
    result, _ = fetch(
        {
            "currency": "USD",
            "periods": [
                {"sessions": [{"id": "task-1", "costCents": amount, "requestCount": 1}]}
            ],
        }
    )
    assert result["status"] == "unavailable"
    assert result["cost_usd"] is None


def test_timeout_is_nonfatal_and_personal_billing_uses_settings():
    with patch("reviewbot.billing.requests.get", side_effect=requests.Timeout) as get:
        result = fetch_session_cost(
            api_key="key",
            bill_to=None,
            session_id="j",
            created_at=1788220800,
            now=1790899200,
        )
    assert result["status"] == "unavailable"
    assert (
        get.call_args.args[0]
        == "https://huggingface.co/api/settings/billing/usage-by-inference-session"
    )
