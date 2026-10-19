"""ESPN-primary NFL injury and posted-starter news, with a Sleeper fallback.

Hits public endpoints only. Missing ESPN and missing Sleeper fail closed: no
invented starter. GSIS IDs are joined from Sleeper when available so v2
qb_prior_epa can use PBP history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Any, Callable, Iterable, Mapping, Sequence
import requests

from src.fantasy.sleeper import SLEEPER_PLAYERS_URL, sleeper_availability
from src.utils.team_codes import canonical_nfl_abbr

GetFn = Callable[..., requests.Response]

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_DEPTH_CHART_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/teams/{team_id}/depthcharts"
)
ESPN_ATHLETE_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/athletes/{athlete_id}"
)
USER_AGENT = "sports-edge-nfl-injury/1.0 (+https://github.com/dmboynton56/sports-edge)"
UNAVAILABLE_STATUSES = {
    "out",
    "ir",
    "injured reserve",
    "doubtful",
    "suspension",
    "sus",
    "susp",
    "pup",
    "dnr",
    "na",
    "inactive",
}
KEY_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}
LINE_MOVE_POINTS = 1.5
ATHLETE_ID_RE = re.compile(r"/athletes/(\d+)")


@dataclass(frozen=True)
class PlayerStatus:
    team: str
    name: str
    position: str
    status: str
    espn_id: str | None = None
    gsis_id: str | None = None
    depth_rank: int | None = None
    comment: str | None = None
    source: str = "espn"


@dataclass(frozen=True)
class PostedStarter:
    team: str
    name: str | None
    gsis_id: str | None
    espn_id: str | None
    depth_rank: int | None
    status: str | None
    skipped: tuple[dict[str, Any], ...] = ()
    unresolved: bool = False
    source: str = "espn"


@dataclass(frozen=True)
class GameNews:
    home_team: str
    away_team: str
    game_date: str | None
    provider_game_id: str | None
    supabase_game_id: str | None
    home_starter: PostedStarter
    away_starter: PostedStarter
    key_injuries: tuple[PlayerStatus, ...]
    espn_home_spread: float | None
    source: str
    fallback_used: bool
    fetched_at: str
    raw_record: dict[str, Any] = field(default_factory=dict)

    @property
    def status_fingerprint(self) -> str:
        return fingerprint_game_news(self)


@dataclass(frozen=True)
class ChangeDecision:
    home_team: str
    away_team: str
    provider_game_id: str | None
    supabase_game_id: str | None
    reasons: tuple[str, ...]
    current: GameNews
    previous_fingerprint: str | None


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def _get(url: str, *, get: GetFn | None = None, timeout: float = 30.0, params: dict[str, Any] | None = None) -> Any:
    fetch = get or requests.get
    response = fetch(url, headers=_headers(), timeout=timeout, params=params)
    response.raise_for_status()
    return response.json()


def _normalize_name(value: object) -> str:
    return " ".join(str(value or "").lower().replace(".", "").replace("'", "").split())


def _normalize_status(value: object) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def athlete_id_from_ref(ref: object) -> str | None:
    match = ATHLETE_ID_RE.search(str(ref or ""))
    return match.group(1) if match else None


def is_unavailable(status: object) -> bool:
    return _normalize_status(status) in UNAVAILABLE_STATUSES


def window_name_for(now: datetime) -> str:
    from zoneinfo import ZoneInfo

    local = now.astimezone(ZoneInfo("America/Denver"))
    weekday, hour = local.weekday(), local.hour
    if weekday == 3 and hour < 12:
        return "thu-am"
    if weekday == 3:
        return "thu-tnf"
    if weekday == 6 and hour < 9:
        return "sun-am"
    if weekday == 6 and hour < 12:
        return "sun-inactives"
    if weekday == 6 and hour < 16:
        return "sun-late"
    if weekday == 6:
        return "sun-snf"
    if weekday == 0:
        return "mon-mnf"
    if weekday == 1 and hour < 12:
        return "tue-am"
    return "adhoc"


def fetch_espn_injuries(*, get: GetFn | None = None, timeout: float = 45.0) -> dict[str, Any]:
    return _get(ESPN_INJURIES_URL, get=get, timeout=timeout)


def fetch_espn_teams(*, get: GetFn | None = None, timeout: float = 30.0) -> list[dict[str, Any]]:
    payload = _get(ESPN_TEAMS_URL, get=get, timeout=timeout, params={"limit": 50})
    sports = payload.get("sports") or []
    leagues = (sports[0].get("leagues") if sports else None) or []
    return list((leagues[0].get("teams") if leagues else None) or payload.get("teams") or [])


def fetch_espn_depth_chart(season: int, team_id: str, *, get: GetFn | None = None) -> dict[str, Any]:
    return _get(ESPN_DEPTH_CHART_URL.format(season=season, team_id=team_id), get=get)


def fetch_espn_scoreboard(dates: str, *, get: GetFn | None = None) -> dict[str, Any]:
    return _get(ESPN_SCOREBOARD_URL, get=get, params={"dates": dates, "limit": 100})


def parse_espn_injuries(payload: Mapping[str, Any]) -> list[PlayerStatus]:
    rows: list[PlayerStatus] = []
    for team in payload.get("injuries") or []:
        team_name = team.get("displayName") or (team.get("team") or {}).get("displayName")
        abbr = canonical_nfl_abbr(team_name) or canonical_nfl_abbr((team.get("team") or {}).get("abbreviation"))
        if not abbr:
            continue
        for item in team.get("injuries") or []:
            athlete = item.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip()
            if not name:
                continue
            position = str(
                (athlete.get("position") or {}).get("abbreviation")
                or (item.get("position") or {}).get("abbreviation")
                or ""
            ).upper()
            type_payload = item.get("type") or {}
            status = _normalize_status(type_payload.get("description") or item.get("status"))
            rows.append(
                PlayerStatus(
                    team=abbr,
                    name=name,
                    position=position,
                    status=status or "active",
                    espn_id=str(athlete.get("id") or "") or None,
                    comment=item.get("shortComment") or item.get("longComment"),
                    source="espn",
                )
            )
    return rows


def parse_espn_team_ids(teams_payload: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ids: dict[str, str] = {}
    for wrapper in teams_payload:
        team = wrapper.get("team") if "team" in wrapper else wrapper
        if not isinstance(team, Mapping):
            continue
        abbr = canonical_nfl_abbr(team.get("abbreviation") or team.get("displayName"))
        team_id = str(team.get("id") or "").strip()
        if abbr and team_id:
            ids[abbr] = team_id
    return ids


def qb_ranks_from_depth_chart(payload: Mapping[str, Any]) -> list[str]:
    """Return ESPN athlete ids for QB ranks 1..n from an offensive depth chart."""
    best: list[tuple[int, str]] = []
    for item in payload.get("items") or []:
        positions = item.get("positions") or {}
        qb = positions.get("qb") or positions.get("QB")
        if not isinstance(qb, Mapping):
            continue
        for athlete in qb.get("athletes") or []:
            athlete_id = athlete_id_from_ref((athlete.get("athlete") or {}).get("$ref"))
            rank = athlete.get("rank")
            try:
                rank_n = int(rank)
            except (TypeError, ValueError):
                continue
            if athlete_id:
                best.append((rank_n, athlete_id))
        if best:
            break
    best.sort()
    seen: list[str] = []
    for _, athlete_id in best:
        if athlete_id not in seen:
            seen.append(athlete_id)
    return seen


def parse_espn_home_spreads(payload: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    spreads: dict[tuple[str, str], float] = {}
    for event in payload.get("events") or []:
        competition = (event.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in competition.get("competitors") or []}
        home = canonical_nfl_abbr(((teams.get("home") or {}).get("team") or {}).get("abbreviation"))
        away = canonical_nfl_abbr(((teams.get("away") or {}).get("team") or {}).get("abbreviation"))
        if not home or not away:
            continue
        odds = (competition.get("odds") or [{}])[0]
        try:
            value = float(odds.get("spread"))
        except (TypeError, ValueError):
            continue
        # ESPN `spread` is typically the home spread (negative = home favored).
        spreads[(away, home)] = value
    return spreads


def sleeper_index(players: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for player in players.values():
        team = canonical_nfl_abbr(player.get("team"))
        name = _normalize_name(player.get("full_name") or "")
        if team and name:
            index[(team, name)] = dict(player)
    return index


def gsis_for(name: str | None, team: str, sleeper_by_name: Mapping[tuple[str, str], Mapping[str, Any]]) -> str | None:
    if not name:
        return None
    player = sleeper_by_name.get((team, _normalize_name(name)))
    if not player:
        return None
    gsis = str(player.get("gsis_id") or "").strip()
    return gsis or None


def sleeper_qb_ranks(players: Mapping[str, Mapping[str, Any]], team: str) -> list[dict[str, Any]]:
    rows = []
    for player in players.values():
        if canonical_nfl_abbr(player.get("team")) != team:
            continue
        if str(player.get("position") or "").upper() != "QB":
            continue
        order = player.get("depth_chart_order")
        try:
            rank = int(order)
        except (TypeError, ValueError):
            continue
        rows.append({"rank": rank, "player": player})
    rows.sort(key=lambda item: item["rank"])
    return rows


def resolve_starter(
    team: str,
    *,
    espn_ranks: Sequence[str],
    injuries: Sequence[PlayerStatus],
    sleeper_players: Mapping[str, Mapping[str, Any]] | None,
    names_by_espn_id: Mapping[str, str],
    source: str = "espn",
) -> PostedStarter:
    by_espn_id = {row.espn_id: row for row in injuries if row.espn_id}
    by_name = {(_normalize_name(row.name), row.team): row for row in injuries}
    sleeper_by_name = sleeper_index(sleeper_players or {})
    skipped: list[dict[str, Any]] = []

    def _status_for(name: str | None, espn_id: str | None) -> str:
        if espn_id and espn_id in by_espn_id:
            return by_espn_id[espn_id].status
        if name:
            hit = by_name.get((_normalize_name(name), team))
            if hit:
                return hit.status
            player = sleeper_by_name.get((team, _normalize_name(name)))
            if player:
                return sleeper_availability(player)
        return "active"

    for rank, espn_id in enumerate(espn_ranks, start=1):
        name = names_by_espn_id.get(espn_id)
        if not name:
            skipped.append({"espn_id": espn_id, "status": "unnamed", "rank": rank})
            continue
        status = _status_for(name, espn_id)
        if is_unavailable(status):
            skipped.append({"name": name, "espn_id": espn_id, "status": status, "rank": rank})
            continue
        return PostedStarter(
            team=team,
            name=name,
            gsis_id=gsis_for(name, team, sleeper_by_name),
            espn_id=espn_id,
            depth_rank=rank,
            status=status,
            skipped=tuple(skipped),
            source=source,
        )

    for item in sleeper_qb_ranks(sleeper_players or {}, team):
        player = item["player"]
        name = str(player.get("full_name") or "").strip()
        status = sleeper_availability(player)
        if is_unavailable(status):
            skipped.append({"name": name, "status": status, "rank": item["rank"], "source": "sleeper"})
            continue
        return PostedStarter(
            team=team,
            name=name or None,
            gsis_id=str(player.get("gsis_id") or "").strip() or None,
            espn_id=None,
            depth_rank=int(item["rank"]),
            status=status,
            skipped=tuple(skipped),
            source="sleeper",
        )

    return PostedStarter(team=team, name=None, gsis_id=None, espn_id=None, depth_rank=None, status=None, skipped=tuple(skipped), unresolved=True, source=source)


def key_injuries_for(teams: Iterable[str], injuries: Sequence[PlayerStatus]) -> tuple[PlayerStatus, ...]:
    wanted = {canonical_nfl_abbr(team) for team in teams}
    rows = []
    for row in injuries:
        if row.team not in wanted:
            continue
        if row.position not in KEY_POSITIONS:
            continue
        unavailable = is_unavailable(row.status)
        qb_questionable = row.position == "QB" and row.status in {"questionable"}
        if unavailable or qb_questionable:
            rows.append(row)
    rows.sort(key=lambda item: (item.team, item.position, item.name))
    return tuple(rows)


def fingerprint_game_news(news: GameNews) -> str:
    payload = {
        "home": [news.home_starter.gsis_id or _normalize_name(news.home_starter.name), news.home_starter.unresolved],
        "away": [news.away_starter.gsis_id or _normalize_name(news.away_starter.name), news.away_starter.unresolved],
        "injuries": [
            [row.team, row.gsis_id or _normalize_name(row.name), row.position, row.status]
            for row in news.key_injuries
        ],
    }
    return sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def detect_changed_games(
    current: Sequence[GameNews],
    previous_by_key: Mapping[str, Mapping[str, Any]],
    *,
    line_move_points: float = LINE_MOVE_POINTS,
) -> list[ChangeDecision]:
    dirty: list[ChangeDecision] = []
    for news in current:
        key = news.supabase_game_id or news.provider_game_id or f"{news.away_team}@{news.home_team}:{news.game_date}"
        prior = previous_by_key.get(str(key))
        reasons: list[str] = []
        if prior is None:
            reasons.append("initial")
        else:
            if str(prior.get("status_fingerprint") or "") != news.status_fingerprint:
                prior_home = prior.get("home_starter_qb_id") or _normalize_name(prior.get("home_starter_qb_name"))
                prior_away = prior.get("away_starter_qb_id") or _normalize_name(prior.get("away_starter_qb_name"))
                curr_home = news.home_starter.gsis_id or _normalize_name(news.home_starter.name)
                curr_away = news.away_starter.gsis_id or _normalize_name(news.away_starter.name)
                if prior_home != curr_home:
                    reasons.append("qb_home")
                if prior_away != curr_away:
                    reasons.append("qb_away")
                if "qb_home" not in reasons and "qb_away" not in reasons:
                    reasons.append("key_injury")
            prior_spread = prior.get("espn_home_spread")
            if (
                news.espn_home_spread is not None
                and prior_spread is not None
                and abs(float(news.espn_home_spread) - float(prior_spread)) + 1e-9 >= line_move_points
            ):
                reasons.append("line_move")
        if reasons:
            dirty.append(
                ChangeDecision(
                    home_team=news.home_team,
                    away_team=news.away_team,
                    provider_game_id=news.provider_game_id,
                    supabase_game_id=news.supabase_game_id,
                    reasons=tuple(reasons),
                    current=news,
                    previous_fingerprint=None if prior is None else str(prior.get("status_fingerprint") or "") or None,
                )
            )
    return dirty


def posted_starters_from_news(news_rows: Sequence[GameNews]) -> dict[tuple[str, str], str]:
    """Map (provider_game_id, team) -> gsis_id for v2 override. Skip unresolved."""
    mapping: dict[tuple[str, str], str] = {}
    for news in news_rows:
        game_id = news.provider_game_id
        if not game_id:
            continue
        if news.home_starter.gsis_id and not news.home_starter.unresolved:
            mapping[(str(game_id), news.home_team)] = news.home_starter.gsis_id
        if news.away_starter.gsis_id and not news.away_starter.unresolved:
            mapping[(str(game_id), news.away_team)] = news.away_starter.gsis_id
    return mapping


def snapshot_row(news: GameNews, *, window_name: str, change_reasons: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "game_id": news.supabase_game_id,
        "provider_game_id": news.provider_game_id,
        "game_date": news.game_date,
        "home_team": news.home_team,
        "away_team": news.away_team,
        "home_starter_qb_id": news.home_starter.gsis_id,
        "home_starter_qb_name": news.home_starter.name,
        "away_starter_qb_id": news.away_starter.gsis_id,
        "away_starter_qb_name": news.away_starter.name,
        "key_injuries": [asdict(row) for row in news.key_injuries],
        "status_fingerprint": news.status_fingerprint,
        "espn_home_spread": news.espn_home_spread,
        "source": news.source,
        "fallback_used": news.fallback_used,
        "fetched_at": news.fetched_at,
        "window_name": window_name,
        "change_reasons": list(change_reasons),
        "raw_record": news.raw_record,
    }


def build_game_news(
    games: Sequence[Mapping[str, Any]],
    *,
    injuries: Sequence[PlayerStatus],
    depth_ranks: Mapping[str, Sequence[str]],
    names_by_espn_id: Mapping[str, str],
    spreads: Mapping[tuple[str, str], float],
    sleeper_players: Mapping[str, Mapping[str, Any]] | None,
    source: str,
    fallback_used: bool,
    fetched_at: datetime,
) -> list[GameNews]:
    rows: list[GameNews] = []
    fetched = fetched_at.astimezone(timezone.utc).isoformat()
    for game in games:
        home = canonical_nfl_abbr(game.get("home_team"))
        away = canonical_nfl_abbr(game.get("away_team"))
        if not home or not away:
            continue
        home_source = "espn" if depth_ranks.get(home) else source
        away_source = "espn" if depth_ranks.get(away) else source
        home_starter = resolve_starter(
            home,
            espn_ranks=depth_ranks.get(home) or (),
            injuries=injuries,
            sleeper_players=sleeper_players,
            names_by_espn_id=names_by_espn_id,
            source=home_source,
        )
        away_starter = resolve_starter(
            away,
            espn_ranks=depth_ranks.get(away) or (),
            injuries=injuries,
            sleeper_players=sleeper_players,
            names_by_espn_id=names_by_espn_id,
            source=away_source,
        )
        game_fallback = fallback_used or home_starter.source == "sleeper" or away_starter.source == "sleeper"
        game_date = game.get("game_date")
        if hasattr(game_date, "isoformat"):
            game_date = game_date.isoformat()
        rows.append(
            GameNews(
                home_team=str(game.get("home_team") or home),
                away_team=str(game.get("away_team") or away),
                game_date=str(game_date)[:10] if game_date else None,
                provider_game_id=str(game.get("provider_game_id") or "") or None,
                supabase_game_id=str(game.get("supabase_game_id") or game.get("id") or game.get("game_id") or "") or None,
                home_starter=home_starter,
                away_starter=away_starter,
                key_injuries=key_injuries_for((home, away), injuries),
                espn_home_spread=spreads.get((away, home)),
                source=source,
                fallback_used=game_fallback,
                fetched_at=fetched,
                raw_record={
                    "home_skipped": list(home_starter.skipped),
                    "away_skipped": list(away_starter.skipped),
                    "home_unresolved": home_starter.unresolved,
                    "away_unresolved": away_starter.unresolved,
                },
            )
        )
    return rows


def resolve_athlete_names(
    athlete_ids: Iterable[str],
    injuries: Sequence[PlayerStatus],
    *,
    season: int,
    get: GetFn | None = None,
) -> dict[str, str]:
    names = {row.espn_id: row.name for row in injuries if row.espn_id and row.name}
    missing = [athlete_id for athlete_id in dict.fromkeys(athlete_ids) if athlete_id not in names]
    for athlete_id in missing:
        try:
            payload = _get(ESPN_ATHLETE_URL.format(season=season, athlete_id=athlete_id), get=get)
        except Exception:  # noqa: BLE001
            continue
        name = str(payload.get("displayName") or payload.get("fullName") or "").strip()
        if name:
            names[athlete_id] = name
    return names


def collect_live_news(
    games: Sequence[Mapping[str, Any]],
    *,
    season: int,
    get: GetFn | None = None,
    sleeper_players: Mapping[str, Mapping[str, Any]] | None = None,
    now: datetime | None = None,
    scoreboard_dates: Sequence[str] = (),
) -> tuple[list[GameNews], dict[str, Any]]:
    """Fetch ESPN, fall back to Sleeper, and build per-game news rows."""
    now = now or datetime.now(timezone.utc)
    summary: dict[str, Any] = {"source": "espn", "fallback_used": False, "espn_error": None, "sleeper_error": None}
    injuries: list[PlayerStatus] = []
    depth_ranks: dict[str, list[str]] = {}
    names_by_espn_id: dict[str, str] = {}
    spreads: dict[tuple[str, str], float] = {}
    try:
        injury_payload = fetch_espn_injuries(get=get)
        injuries = parse_espn_injuries(injury_payload)
        team_ids = parse_espn_team_ids(fetch_espn_teams(get=get))
        needed = {
            abbr
            for game in games
            for abbr in (canonical_nfl_abbr(game.get("home_team")), canonical_nfl_abbr(game.get("away_team")))
            if abbr
        }
        for abbr in sorted(needed):
            team_id = team_ids.get(abbr)
            if not team_id:
                continue
            try:
                depth_ranks[abbr] = qb_ranks_from_depth_chart(fetch_espn_depth_chart(season, team_id, get=get))
            except Exception as exc:  # noqa: BLE001
                summary.setdefault("depth_errors", {})[abbr] = str(exc)
        athlete_ids = [athlete_id for ranks in depth_ranks.values() for athlete_id in ranks]
        names_by_espn_id = resolve_athlete_names(athlete_ids, injuries, season=season, get=get)
        for dates in scoreboard_dates:
            try:
                spreads.update(parse_espn_home_spreads(fetch_espn_scoreboard(dates, get=get)))
            except Exception as exc:  # noqa: BLE001
                summary.setdefault("scoreboard_errors", {})[dates] = str(exc)
    except Exception as exc:  # noqa: BLE001
        summary["espn_error"] = str(exc)
        summary["source"] = "sleeper"
        summary["fallback_used"] = True

    if summary["fallback_used"] and sleeper_players is None:
        try:
            sleeper_players = _get(SLEEPER_PLAYERS_URL, get=get, timeout=45.0)
            if not isinstance(sleeper_players, dict):
                raise ValueError("Sleeper returned a non-object player directory")
        except Exception as exc:  # noqa: BLE001
            summary["sleeper_error"] = str(exc)
            sleeper_players = None

    if not injuries and sleeper_players:
        for player in sleeper_players.values():
            team = canonical_nfl_abbr(player.get("team"))
            name = str(player.get("full_name") or "").strip()
            if not team or not name:
                continue
            status = sleeper_availability(player)
            if status == "expected":
                continue
            injuries.append(
                PlayerStatus(
                    team=team,
                    name=name,
                    position=str(player.get("position") or "").upper(),
                    status=status,
                    gsis_id=str(player.get("gsis_id") or "").strip() or None,
                    depth_rank=int(player["depth_chart_order"]) if str(player.get("depth_chart_order") or "").isdigit() else None,
                    source="sleeper",
                )
            )

    if summary["espn_error"] and not sleeper_players:
        raise RuntimeError(f"NFL injury news failed closed (ESPN and Sleeper): {summary['espn_error']}")

    if sleeper_players is None and not summary["fallback_used"]:
        # ID enrichment only; ESPN remains the status authority.
        try:
            sleeper_players = _get(SLEEPER_PLAYERS_URL, get=get, timeout=45.0)
            if not isinstance(sleeper_players, dict):
                sleeper_players = None
        except Exception as exc:  # noqa: BLE001
            summary["sleeper_error"] = str(exc)
            sleeper_players = None

    news = build_game_news(
        games,
        injuries=injuries,
        depth_ranks=depth_ranks,
        names_by_espn_id=names_by_espn_id,
        spreads=spreads,
        sleeper_players=sleeper_players,
        source=str(summary["source"]),
        fallback_used=bool(summary["fallback_used"]),
        fetched_at=now,
    )
    summary["games"] = len(news)
    summary["unresolved_starters"] = sum(
        int(row.home_starter.unresolved) + int(row.away_starter.unresolved) for row in news
    )
    return news, summary
