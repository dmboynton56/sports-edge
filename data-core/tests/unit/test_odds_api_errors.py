"""Tests for The Odds API quota/auth exhaustion classification."""

from __future__ import annotations

import json

import pytest
import requests

from src.data.odds_api_errors import (
    OddsApiQuotaExhausted,
    check_odds_api_response,
    is_odds_api_quota_error,
    is_odds_api_quota_failure,
)


DAILY_300_BODY = json.dumps(
    {
        "message": "Usage quota has been reached. See usage plans at https://the-odds-api.com",
        "error_code": "OUT_OF_USAGE_CREDITS",
        "details_url": "https://the-odds-api.com/liveapi/guides/v4/api-error-codes.html#out-of-usage-credits",
    }
)


class FakeResponse:
    def __init__(self, status_code, text="", headers=None, json_data=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self._json = json_data

    def json(self):
        if self._json is not None:
            return self._json
        return json.loads(self.text)


def test_out_of_usage_credits_payload_is_quota():
    assert is_odds_api_quota_failure(
        status_code=401,
        body=DAILY_300_BODY,
        error_code="OUT_OF_USAGE_CREDITS",
    )


def test_429_is_quota_even_without_body():
    assert is_odds_api_quota_failure(status_code=429, body="")


def test_zero_remaining_on_failed_request_is_quota():
    assert is_odds_api_quota_failure(status_code=401, body="", requests_remaining="0")


def test_zero_remaining_on_success_is_not_classified_as_failure():
    assert is_odds_api_quota_failure(status_code=200, body="[]", requests_remaining="0") is False


def test_invalid_api_key_401_is_not_quota():
    body = json.dumps({"message": "API key is invalid", "error_code": "INVALID_API_KEY"})
    assert is_odds_api_quota_failure(status_code=401, body=body, error_code="INVALID_API_KEY") is False


def test_500_is_not_quota():
    assert is_odds_api_quota_failure(status_code=500, body="upstream exploded") is False


def test_timeout_exception_is_hard_fail():
    assert is_odds_api_quota_error(requests.Timeout("timed out")) is False


def test_quota_runtimeerror_from_daily_300_is_soft():
    exc = RuntimeError(f"Error fetching NFL odds from API: 401 {DAILY_300_BODY}")
    assert is_odds_api_quota_error(exc)


def test_check_odds_api_response_raises_quota_on_out_of_usage():
    response = FakeResponse(401, DAILY_300_BODY)
    with pytest.raises(OddsApiQuotaExhausted, match="OUT_OF_USAGE_CREDITS|Usage quota"):
        check_odds_api_response(response, context="NFL odds")


def test_check_odds_api_response_raises_runtimeerror_on_500():
    response = FakeResponse(500, "upstream exploded")
    with pytest.raises(RuntimeError, match="500"):
        check_odds_api_response(response, context="NFL odds")
    with pytest.raises(RuntimeError) as caught:
        check_odds_api_response(response, context="NFL odds")
    assert not isinstance(caught.value, OddsApiQuotaExhausted)


def test_check_odds_api_response_accepts_200():
    check_odds_api_response(FakeResponse(200, "[]", json_data=[]))
