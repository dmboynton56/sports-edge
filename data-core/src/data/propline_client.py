"""PropLine API client for MLB player prop odds.

PropLine provides MLB player prop market odds including home runs.
API docs: https://prop-line.com/llms-full.txt
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

LOG = logging.getLogger(__name__)

PROPLINE_BASE_URL = "https://api.prop-line.com/v1"
PROPLINE_SPORT_KEY = "baseball_mlb"


class PropLineError(RuntimeError):
    """Raised when PropLine API returns a non-recoverable response."""


@dataclass
class PropLineResponseMeta:
    status_code: int = 0
    rate_limit_remaining: Optional[str] = None
    rate_limit_reset: Optional[str] = None


@dataclass
class PropLineClient:
    """PropLine API client with rate-limiting and header auth support."""

    api_key: str
    timeout: int = 30
    min_request_interval_sec: float = 0.25
    request_count: int = 0
    response_meta: PropLineResponseMeta = field(default_factory=PropLineResponseMeta)
    _last_request_at: float = 0.0

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Execute GET request to PropLine API.

        PropLine accepts API key via query param apiKey= or header X-API-Key.
        We use X-API-Key header to avoid logging keys in URLs.
        """
        elapsed = time.time() - self._last_request_at
        if elapsed < self.min_request_interval_sec:
            time.sleep(self.min_request_interval_sec - elapsed)

        headers = {"X-API-Key": self.api_key}
        url = f"{PROPLINE_BASE_URL}{path}"
        response = requests.get(url, headers=headers, params=params or {}, timeout=self.timeout)
        self._last_request_at = time.time()
        self.request_count += 1
        self.response_meta = PropLineResponseMeta(
            status_code=response.status_code,
            rate_limit_remaining=response.headers.get("X-RateLimit-Remaining"),
            rate_limit_reset=response.headers.get("X-RateLimit-Reset"),
        )

        if response.status_code >= 400:
            try:
                payload = response.json()
                message = payload.get("message") or payload.get("error") or response.text
            except ValueError:
                message = response.text
            raise PropLineError(f"PropLine API {response.status_code}: {message}")

        if not response.text:
            return {}
        return response.json()


def get_propline_api_key() -> str:
    """Load PropLine API key from environment."""
    key = os.getenv("PROPLINE_API_KEY")
    if not key:
        raise ValueError("PROPLINE_API_KEY not found in environment")
    return key


def fetch_propline_mlb_events(client: PropLineClient) -> list[dict[str, Any]]:
    """Fetch all active MLB events from PropLine."""
    payload = client.get(f"/sports/{PROPLINE_SPORT_KEY}/events")
    return payload if isinstance(payload, list) else []


def fetch_propline_event_odds(
    client: PropLineClient,
    *,
    event_id: str,
    markets: str | list[str],
) -> dict[str, Any]:
    """Fetch odds for a specific PropLine event.

    Args:
        client: PropLine API client
        event_id: PropLine event ID
        markets: Comma-separated string or list of market keys (e.g., "batter_home_runs")

    Returns:
        Event odds payload compatible with The Odds API format
    """
    if isinstance(markets, list):
        markets_str = ",".join(markets)
    else:
        markets_str = markets

    payload = client.get(
        f"/sports/{PROPLINE_SPORT_KEY}/events/{event_id}/odds",
        {"markets": markets_str},
    )
    return payload if isinstance(payload, dict) else {}
