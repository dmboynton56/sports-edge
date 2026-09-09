"""Classify non-transient The Odds API quota / usage failures.

Starter-plan keys return HTTP 401 with error_code OUT_OF_USAGE_CREDITS when
the monthly credit pool is empty. Retrying that response cannot succeed until
the monthly reset, so callers should skip Odds writes (fail closed: no invented
prices) and let the rest of Daily Refresh continue.
"""

from __future__ import annotations

import re
from typing import Any

QUOTA_ERROR_CODES = frozenset(
    {
        "OUT_OF_USAGE_CREDITS",
        "OUT_OF_USAGE",
    }
)

_QUOTA_PHRASES = (
    "out_of_usage_credits",
    "out of usage credits",
    "usage quota",
    "quota has been reached",
    "quota exhausted",
    "insufficient credits",
    "no requests remaining",
)


class OddsApiQuotaExhausted(RuntimeError):
    """Quota, usage, or rate-limit exhaustion that will not recover this run."""


def _normalize(text: str | None) -> str:
    return (text or "").lower()


def body_has_quota_language(body: str | None) -> bool:
    text = _normalize(body)
    return any(phrase in text for phrase in _QUOTA_PHRASES)


def remaining_credits_exhausted(requests_remaining: str | int | None) -> bool:
    if requests_remaining is None:
        return False
    raw = str(requests_remaining).strip()
    try:
        return int(raw) == 0
    except (TypeError, ValueError):
        return raw == "0"


def is_odds_api_quota_failure(
    *,
    status_code: int | None = None,
    body: str | None = None,
    error_code: str | None = None,
    requests_remaining: str | int | None = None,
) -> bool:
    """True for quota/auth-exhaustion/rate-limit responses that should not retry."""
    if error_code and str(error_code).strip().upper() in QUOTA_ERROR_CODES:
        return True
    if body_has_quota_language(body):
        return True
    if status_code == 429:
        return True
    if remaining_credits_exhausted(requests_remaining) and status_code not in (None, 200):
        return True
    return False


def is_odds_api_quota_error(exc: BaseException) -> bool:
    """True for OddsApiQuotaExhausted or exception text that matches quota/429."""
    if isinstance(exc, OddsApiQuotaExhausted):
        return True
    text = str(exc)
    if is_odds_api_quota_failure(body=text, error_code=_error_code_from_text(text)):
        return True
    return bool(re.search(r"\b429\b", text))


def _error_code_from_text(text: str) -> str | None:
    match = re.search(r"OUT_OF_USAGE_CREDITS|OUT_OF_USAGE\b", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def check_odds_api_response(response: Any, *, context: str = "The Odds API") -> None:
    """Raise OddsApiQuotaExhausted or RuntimeError for a non-200 Odds API response.

    200 responses return immediately. Network failures never reach this helper.
    """
    status = getattr(response, "status_code", None)
    if status == 200:
        return

    headers = getattr(response, "headers", None) or {}
    remaining = headers.get("x-requests-remaining")
    if remaining is None:
        remaining = headers.get("X-Requests-Remaining")

    body = getattr(response, "text", "") or ""
    error_code = None
    json_fn = getattr(response, "json", None)
    if callable(json_fn):
        try:
            payload = json_fn()
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict):
            error_code = payload.get("error_code")
            body = payload.get("message") or payload.get("error") or body

    detail = f"{context}: {status} {body}".strip()
    if is_odds_api_quota_failure(
        status_code=status,
        body=str(body),
        error_code=None if error_code is None else str(error_code),
        requests_remaining=remaining,
    ):
        raise OddsApiQuotaExhausted(detail)
    raise RuntimeError(f"Error fetching odds from API: {detail}")
