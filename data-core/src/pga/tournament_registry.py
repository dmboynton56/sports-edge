"""Registry loading and path helpers for PGA tournament automation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.pga.live_leaderboard import event_matches


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
DEFAULT_REGISTRY_PATH = ROOT / "config" / "pga_tournaments.yaml"


@dataclass(frozen=True)
class PgaTournament:
    key: str
    name: str
    season: int
    course: str
    par: int
    start_date: date
    end_date: date
    yardage: int | None = None
    odds_key: str | None = None
    espn_event_id: str | None = None
    espn_match: tuple[str, ...] = ()
    total_rounds: int = 4
    cut_after_round: int = 2
    cut_size: int = 65
    cut_rule: str = "top_n_and_ties"
    field_source: str | None = None
    field_fetcher: str | None = None
    prediction_window_days: int = 4
    post_window_days: int = 2
    priority: int = 0

    @property
    def field_json(self) -> Path:
        return ROOT / "src" / "data" / "fields" / f"{self.key}_field.json"

    @property
    def field_text(self) -> Path:
        return ROOT / "src" / "data" / "fields" / f"{self.key}_field.txt"

    @property
    def predictions_csv(self) -> Path:
        return ROOT / "notebooks" / "cache" / f"{self.key}_predictions.csv"

    @property
    def midtournament_csv(self) -> Path:
        return ROOT / "notebooks" / "cache" / f"{self.key}_midtournament.csv"

    @property
    def public_json(self) -> Path:
        return REPO_ROOT / "web" / "public" / "data" / "pga_tournaments" / f"{self.key}.json"

    @property
    def current_json(self) -> Path:
        return REPO_ROOT / "web" / "public" / "data" / "pga_tournaments" / "current.json"


@dataclass(frozen=True)
class PgaRegistry:
    season: int
    tournaments: tuple[PgaTournament, ...]

    def by_key(self, key: str) -> PgaTournament:
        for tournament in self.tournaments:
            if tournament.key == key:
                return tournament
        known = ", ".join(t.key for t in self.tournaments)
        raise KeyError(f"Unknown PGA tournament key {key!r}. Known keys: {known}")


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> PgaRegistry:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    season = int(payload.get("season") or datetime.now().year)
    tournaments = []
    for raw in payload.get("tournaments") or []:
        start = _parse_date(raw["start_date"])
        end = _parse_date(raw["end_date"])
        tournaments.append(
            PgaTournament(
                key=str(raw["key"]),
                name=str(raw.get("name") or raw["key"]),
                season=int(raw.get("season") or season),
                course=str(raw.get("course") or ""),
                par=int(raw.get("par") or 72),
                yardage=int(raw["yardage"]) if raw.get("yardage") is not None else None,
                odds_key=str(raw.get("odds_key") or raw["key"]),
                espn_event_id=str(raw["espn_event_id"]) if raw.get("espn_event_id") else None,
                espn_match=_as_tuple(raw.get("espn_match")),
                start_date=start,
                end_date=end,
                total_rounds=int(raw.get("total_rounds") or 4),
                cut_after_round=int(raw.get("cut_after_round") or 2),
                cut_size=int(raw.get("cut_size") or 65),
                cut_rule=str(raw.get("cut_rule") or "top_n_and_ties"),
                field_source=str(raw["field_source"]) if raw.get("field_source") else None,
                field_fetcher=str(raw["field_fetcher"]) if raw.get("field_fetcher") else None,
                prediction_window_days=int(raw.get("prediction_window_days") or 4),
                post_window_days=int(raw.get("post_window_days") or 2),
                priority=int(raw.get("priority") or 0),
            )
        )
    return PgaRegistry(season=season, tournaments=tuple(tournaments))


def _as_date(value: date | datetime | str | None) -> date:
    if value is None:
        return datetime.utcnow().date()
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    if isinstance(value, datetime):
        return value.date()
    return value


def resolve_active_tournament(
    registry: PgaRegistry,
    *,
    tournament_key: str | None = None,
    as_of: date | datetime | str | None = None,
    scoreboard: dict[str, Any] | None = None,
) -> PgaTournament | None:
    """Resolve the event whose automation window includes ``as_of``."""

    if tournament_key:
        return registry.by_key(tournament_key)

    anchor = _as_date(as_of)
    events = (scoreboard or {}).get("events") or []
    candidates: list[tuple[int, int, int, int, PgaTournament]] = []
    for tournament in registry.tournaments:
        pre_start = tournament.start_date - timedelta(days=tournament.prediction_window_days)
        post_end = tournament.end_date + timedelta(days=tournament.post_window_days)
        if not (pre_start <= anchor <= post_end):
            continue
        if tournament.start_date <= anchor <= tournament.end_date:
            phase_rank = 3
            distance = 0
        elif anchor < tournament.start_date:
            phase_rank = 2
            distance = (tournament.start_date - anchor).days
        else:
            phase_rank = 1
            distance = (anchor - tournament.end_date).days
        match_score = max((event_matches(event, tournament.espn_match) for event in events), default=0)
        candidates.append((phase_rank, match_score, tournament.priority, -distance, tournament))

    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row[:4], reverse=True)[0][4]


def infer_phase(
    tournament: PgaTournament,
    *,
    as_of: date | datetime | str | None = None,
    leaderboard: dict[str, Any] | None = None,
    force_phase: str | None = None,
) -> str:
    """Infer refresh phase: pre, live, or post."""

    if force_phase:
        phase = force_phase.strip().lower()
        if phase not in {"pre", "live", "post"}:
            raise ValueError("--force-phase must be one of pre, live, post")
        return phase

    anchor = _as_date(as_of)
    if anchor < tournament.start_date:
        return "pre"
    if leaderboard and leaderboard.get("isCompleted") and int(leaderboard.get("currentRound") or 0) >= tournament.total_rounds:
        return "post"
    if anchor <= tournament.end_date:
        return "live"
    return "post"


def event_status_for_phase(phase: str) -> str:
    return {
        "pre": "pre_tournament",
        "live": "in_progress",
        "post": "completed",
    }.get(phase, phase)
