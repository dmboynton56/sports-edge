"""Small, server-side FantasyPros API client.

The API key is intentionally read only from the environment. The web client
never calls FantasyPros directly; refresh jobs cache normalized rows for the
public projection board. The parameter names mirror the documented v2 API so
the adapter can be reused for rankings, ADP, projections, player metadata,
news, injuries, and historical points.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping

import requests


DEFAULT_BASE_URL = "https://api.fantasypros.com/public/v2/json"


class FantasyProsError(RuntimeError):
    """Raised when a FantasyPros request cannot be completed."""


@dataclass(frozen=True)
class FantasyProsClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "FantasyProsClient":
        key = os.getenv("FANTASYPROS_API_KEY") or os.getenv("FANTASY_PROS_API_KEY")
        if not key:
            raise FantasyProsError("FANTASYPROS_API_KEY is not configured")
        return cls(api_key=key)

    def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        response = requests.get(
            url,
            headers={"x-api-key": self.api_key, "Accept": "application/json"},
            params={key: value for key, value in (params or {}).items() if value not in (None, "")},
            timeout=self.timeout,
        )
        if not response.ok:
            detail = response.text[:500].replace("\n", " ")
            raise FantasyProsError(f"FantasyPros {response.status_code} for {url}: {detail}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise FantasyProsError(f"FantasyPros returned a non-object payload for {url}")
        return payload

    def players(
        self,
        *,
        sport: str = "nfl",
        player: str | int | None = None,
        updated: date | str | None = None,
        show: str | None = None,
    ) -> dict[str, Any]:
        return self.get(
            f"{sport}/players",
            {"player": player, "updated": str(updated) if updated else None, "show": show},
        )

    def consensus_rankings(
        self,
        *,
        season: int,
        position: str = "ALL",
        scoring: str = "PPR",
        ranking_type: str | None = None,
        week: int = 0,
        experts: bool | None = None,
        filters: str | Iterable[str] | None = None,
        sport: str = "nfl",
    ) -> dict[str, Any]:
        if isinstance(filters, (tuple, list, set)):
            filters = ":".join(str(value) for value in filters)
        return self.get(
            f"{sport}/{season}/consensus-rankings",
            {
                "position": position,
                "scoring": scoring,
                "type": ranking_type,
                "week": week,
                # The v2 API uses the ``experts`` query input to opt into
                # expert metadata.  ``filters`` remains the separate
                # colon-delimited expert allow-list.
                "experts": str(experts).lower() if experts is not None else None,
                "filters": filters,
            },
        )

    def projections(
        self,
        *,
        season: int,
        week: int = 0,
        position: str | None = None,
        positions: Iterable[str] | str | None = None,
        scoring: str = "PPR",
        players: Iterable[str | int] | str | None = None,
        filters: Iterable[str] | str | None = None,
        sport: str = "nfl",
    ) -> dict[str, Any]:
        def colon_join(value: Iterable[str | int] | str | None) -> str | None:
            if value is None or isinstance(value, str):
                return value
            return ":".join(str(item) for item in value)

        return self.get(
            f"{sport}/{season}/projections",
            {
                "week": week,
                "position": position,
                "positions": colon_join(positions),
                "scoring": scoring,
                "players": colon_join(players),
                "filters": colon_join(filters),
            },
        )

    def player_points(
        self,
        *,
        season: int,
        start: int | None = None,
        end: int | None = None,
        position: str = "ALL",
        scoring: str = "PPR",
    ) -> dict[str, Any]:
        return self.get(
            f"nfl/{season}/player-points",
            {"start": start, "end": end, "position": position, "scoring": scoring},
        )

    def news(
        self,
        *,
        fpid: str | int | None = None,
        limit: int = 25,
        category: str | None = None,
        sport: str = "nfl",
    ) -> dict[str, Any]:
        return self.get(f"{sport}/news", {"fpid": fpid, "limit": limit, "category": category})

    def injuries(
        self,
        *,
        season: int | None = None,
        week: int | None = None,
        sport: str = "nfl",
    ) -> dict[str, Any]:
        return self.get(f"{sport}/injuries", {"season": season, "week": week})


def normalize_consensus_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize consensus/ADP data while retaining provider provenance."""

    rows: list[dict[str, Any]] = []
    for row in payload.get("players", []) or []:
        if not isinstance(row, Mapping):
            continue
        # The v2 endpoint emits ``rank_ecr`` for ordinary consensus and may
        # reuse that field when ``type=ADP`` is requested. Preserve both the
        # explicit ADP field and the effective rank so the caller can label it.
        rank_adp = row.get("rank_adp") or row.get("adp") or row.get("rank_ecr")
        rank_ecr = row.get("rank_ecr")
        rows.append(
            {
                "provider": "fantasypros",
                "player_id": row.get("player_id") or row.get("fpid"),
                "player_name": row.get("player_name") or row.get("name"),
                "team": row.get("player_team_id") or row.get("team_id"),
                "position": row.get("player_position_id") or row.get("position_id"),
                "position_rank": row.get("pos_rank"),
                "adp": _float(rank_adp),
                "consensus_rank": _float(rank_ecr),
                "consensus_min": _float(row.get("rank_min")),
                "consensus_max": _float(row.get("rank_max")),
                "consensus_std": _float(row.get("rank_std")),
                "tier": _int(row.get("tier")),
                "last_updated": payload.get("last_updated"),
            }
        )
    return rows


def normalize_projection_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("players", []) or payload.get("player", []) or []:
        if not isinstance(row, Mapping):
            continue
        stats = row.get("stats") if isinstance(row.get("stats"), Mapping) else row
        rows.append(
            {
                "provider": "fantasypros",
                "player_id": row.get("fpid") or row.get("player_id"),
                "player_name": row.get("name") or row.get("player_name"),
                "team": row.get("team_id") or row.get("player_team_id"),
                "position": row.get("position_id") or row.get("player_position_id"),
                "stats": {str(key): _float(value) for key, value in stats.items() if _float(value) is not None},
                "season": payload.get("season"),
                "week": payload.get("week"),
            }
        )
    return rows


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None
