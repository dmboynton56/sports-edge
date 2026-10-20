#!/usr/bin/env python3
"""Read-only NFL v2 live forecasts for the upcoming slate.

Uses the frozen ``nfl-v2-live-20260906`` artifact, leakage-safe nflverse
pregame features, ESPN/Sleeper posted-QB overrides, and free ESPN DK lines.
This is the no-credentials stand-in for Daily Refresh ``refresh_nfl``.
Does not write BigQuery or Supabase.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_nfl_2026_week7 import load_team_games
from src.data.espn_dk_odds import ESPN_DK_BOOK, fetch_espn_dk_events
from src.data.nfl_injury_news import collect_live_news, posted_starters_from_news
from src.fantasy.sleeper import load_nflverse_rosters, load_sleeper_players
from src.models.nfl_live import LIVE_MODEL_VERSION, predict_live_games
from src.models.nfl_toss_up import is_nfl_toss_up
from src.models.nfl_v2 import no_vig_home_probability
from src.utils.nfl_schedule import DENVER
from src.utils.team_codes import canonical_nfl_abbr

CACHE_DIR = ROOT / "notebooks" / "cache"
DEFAULT_OUTPUT = CACHE_DIR / "nfl_2026_upcoming_predictions.json"
DEFAULT_START = "2026-10-22"
DEFAULT_END = "2026-10-26"
BYES = ("BUF", "LAC", "WAS", "JAX")


def _to_pandas(value) -> pd.DataFrame:
    if hasattr(value, "collect"):
        value = value.collect()
    return value.to_pandas() if hasattr(value, "to_pandas") else pd.DataFrame(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--week", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def kickoff_mt(game_date: object, gametime: object, espn_commence: str | None) -> str | None:
    if espn_commence:
        try:
            stamp = datetime.fromisoformat(espn_commence.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            return stamp.astimezone(DENVER).strftime("%Y-%m-%d %H:%M %Z")
        except ValueError:
            pass
    if game_date is None:
        return None
    day = pd.Timestamp(game_date).strftime("%Y-%m-%d")
    clock = str(gametime or "13:00")
    try:
        eastern = datetime.strptime(f"{day} {clock}", "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))
    except ValueError:
        return f"{day} {clock} ET"
    return eastern.astimezone(DENVER).strftime("%Y-%m-%d %H:%M %Z")


def _norm_name(value: object) -> str:
    return " ".join(str(value or "").lower().replace(".", "").replace("'", "").split())


def index_nflverse_gsis(rosters: dict[str, dict[str, Any]]) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    by_espn: dict[str, str] = {}
    by_name_team: dict[tuple[str, str], str] = {}
    for gsis, record in rosters.items():
        espn_id = str(record.get("espn_id") or "").strip()
        if espn_id:
            by_espn[espn_id] = gsis
        team = canonical_nfl_abbr(record.get("team"))
        name = _norm_name(record.get("full_name") or record.get("player_name"))
        if team and name:
            by_name_team[(team, name)] = gsis
    return by_espn, by_name_team


def starter_gsis(starter, team: str, by_espn: dict[str, str], by_name_team: dict[tuple[str, str], str]) -> str | None:
    if starter.gsis_id:
        return starter.gsis_id
    if starter.espn_id and starter.espn_id in by_espn:
        return by_espn[starter.espn_id]
    team_code = canonical_nfl_abbr(team) or team
    if starter.name:
        hit = by_name_team.get((team_code, _norm_name(starter.name)))
        if hit:
            return hit
    return None


def posted_starters_with_rosters(news_rows, rosters: dict[str, dict[str, Any]]) -> dict[tuple[str, str], str]:
    mapping = dict(posted_starters_from_news(news_rows))
    by_espn, by_name_team = index_nflverse_gsis(rosters)
    for row in news_rows:
        game_id = row.provider_game_id
        if not game_id:
            continue
        for team, starter in ((row.home_team, row.home_starter), (row.away_team, row.away_starter)):
            if starter.unresolved:
                continue
            gsis = starter_gsis(starter, team, by_espn, by_name_team)
            team_code = canonical_nfl_abbr(team) or team
            if gsis:
                mapping[(str(game_id), team_code)] = gsis
                mapping[(str(game_id), str(team))] = gsis
    return mapping


def qb_note(starter) -> str:
    skipped = []
    for item in starter.skipped:
        name = item.get("name") or item.get("espn_id") or "unnamed"
        status = item.get("status") or "unavailable"
        skipped.append(f"{name} ({status})")
    if starter.unresolved:
        skipped_text = f"; skipped {', '.join(skipped)}" if skipped else ""
        return f"unresolved posted starter{skipped_text}; last-3 QB mode"
    if skipped:
        return f"posted {starter.name} after skipping {', '.join(skipped)}"
    if starter.name:
        return f"posted {starter.name} (depth {starter.depth_rank}, {starter.source})"
    return "last-3 QB mode"


def dk_lookup(events: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        home = canonical_nfl_abbr(event.get("home_abbr") or event.get("home_team"))
        away = canonical_nfl_abbr(event.get("away_abbr") or event.get("away_team"))
        if not home or not away:
            continue
        book = (event.get("bookmakers") or [{}])[0]
        markets = {market.get("key"): market for market in book.get("markets") or []}
        spread_market = markets.get("spreads") or {}
        ml_market = markets.get("h2h") or {}
        home_spread = next(
            (
                outcome
                for outcome in spread_market.get("outcomes") or []
                if canonical_nfl_abbr(outcome.get("name")) == home
            ),
            None,
        )
        home_ml = next(
            (
                outcome
                for outcome in ml_market.get("outcomes") or []
                if canonical_nfl_abbr(outcome.get("name")) == home
            ),
            None,
        )
        away_ml = next(
            (
                outcome
                for outcome in ml_market.get("outcomes") or []
                if canonical_nfl_abbr(outcome.get("name")) == away
            ),
            None,
        )
        index[(away, home)] = {
            "dk_spread": None if home_spread is None else float(home_spread["point"]),
            "dk_home_ml": None if home_ml is None else int(home_ml["price"]),
            "dk_away_ml": None if away_ml is None else int(away_ml["price"]),
            "dk_no_vig_home_pct": no_vig_home_probability(
                None if home_ml is None else home_ml.get("price"),
                None if away_ml is None else away_ml.get("price"),
            ),
            "commence_time": event.get("commence_time"),
            "provider_event_id": event.get("id"),
            "book": ESPN_DK_BOOK,
        }
    return index


def main() -> None:
    args = _parse_args()
    import nflreadpy as nfl

    seasons = list(range(2020, args.season + 1))
    print("Loading nflverse schedules...", flush=True)
    schedules = _to_pandas(nfl.load_schedules(seasons))
    schedules["league"] = "NFL"
    team_games = load_team_games(seasons)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    game_dates = pd.to_datetime(schedules.get("gameday", schedules.get("game_date")), errors="coerce")
    upcoming = schedules[
        (pd.to_numeric(schedules["season"], errors="coerce") == args.season)
        & (game_dates >= start)
        & (game_dates <= end)
        & (schedules["home_score"].isna())
        & (schedules["away_score"].isna())
    ].copy()
    if upcoming.empty:
        upcoming = schedules[
            (pd.to_numeric(schedules["season"], errors="coerce") == args.season)
            & (pd.to_numeric(schedules["week"], errors="coerce") == args.week)
            & (schedules["home_score"].isna())
        ].copy()
    target_ids = [str(value) for value in upcoming["game_id"].tolist()]
    if not target_ids:
        raise SystemExit(f"No unplayed NFL games between {args.start_date} and {args.end_date}")

    news_games = []
    for row in upcoming.itertuples(index=False):
        news_games.append(
            {
                "home_team": canonical_nfl_abbr(row.home_team) or row.home_team,
                "away_team": canonical_nfl_abbr(row.away_team) or row.away_team,
                "game_date": str(getattr(row, "gameday", getattr(row, "game_date", "")))[:10],
                "provider_game_id": str(row.game_id),
            }
        )
    sleeper = None
    sleeper_error = None
    try:
        sleeper = load_sleeper_players()
    except Exception as exc:  # noqa: BLE001
        sleeper_error = str(exc)
    print("Fetching ESPN injuries + depth charts...", flush=True)
    news, news_summary = collect_live_news(
        news_games,
        season=args.season,
        sleeper_players=sleeper,
        now=datetime.now(timezone.utc),
        scoreboard_dates=[start.strftime("%Y%m%d"), end.strftime("%Y%m%d")],
    )
    try:
        rosters = load_nflverse_rosters(args.season)
    except Exception as exc:  # noqa: BLE001
        rosters = {}
        news_summary["nflverse_roster_error"] = str(exc)
    posted = posted_starters_with_rosters(news, rosters)
    by_espn, by_name_team = index_nflverse_gsis(rosters)
    news_by_game = {str(row.provider_game_id): row for row in news if row.provider_game_id}

    print(f"Scoring {len(target_ids)} games with {LIVE_MODEL_VERSION}...", flush=True)
    predictions = predict_live_games(schedules, team_games, target_ids, posted_starters=posted or None)

    print("Fetching ESPN DK odds...", flush=True)
    try:
        dk_events = fetch_espn_dk_events("NFL", week=args.week, season_type=2)
    except Exception as exc:  # noqa: BLE001
        dk_events = []
        news_summary["espn_dk_error"] = str(exc)
    dk_by_matchup = dk_lookup(dk_events)

    games_out: list[dict[str, Any]] = []
    for row in predictions.sort_values(["game_date", "away_team"]).itertuples(index=False):
        home = canonical_nfl_abbr(row.home_team) or str(row.home_team)
        away = canonical_nfl_abbr(row.away_team) or str(row.away_team)
        news_row = news_by_game.get(str(row.game_id))
        dk = dk_by_matchup.get((away, home), {})
        home_win = float(row.home_win_probability)
        model_spread = float(row.predicted_spread)
        dk_spread = dk.get("dk_spread")
        dk_home_pct = dk.get("dk_no_vig_home_pct")
        schedule_row = upcoming.loc[upcoming["game_id"].astype(str) == str(row.game_id)].iloc[0]
        games_out.append(
            {
                "game_id": str(row.game_id),
                "season": int(row.season),
                "week": int(row.season_week) if pd.notna(row.season_week) else args.week,
                "away_team": away,
                "home_team": home,
                "kickoff_mt": kickoff_mt(
                    schedule_row.get("gameday", row.game_date),
                    schedule_row.get("gametime"),
                    dk.get("commence_time"),
                ),
                "model_spread": round(model_spread, 2),
                "home_win_prob": round(home_win, 4),
                "toss_up": bool(is_nfl_toss_up(home_win)),
                "home_qb": None if news_row is None else news_row.home_starter.name,
                "away_qb": None if news_row is None else news_row.away_starter.name,
                "home_qb_id": None
                if news_row is None
                else starter_gsis(news_row.home_starter, home, by_espn, by_name_team),
                "away_qb_id": None
                if news_row is None
                else starter_gsis(news_row.away_starter, away, by_espn, by_name_team),
                "home_qb_note": None if news_row is None else qb_note(news_row.home_starter),
                "away_qb_note": None if news_row is None else qb_note(news_row.away_starter),
                "qb_override_applied": bool(
                    posted.get((str(row.game_id), home)) or posted.get((str(row.game_id), away))
                ),
                "dk_spread": dk_spread,
                "dk_no_vig_home_pct": None if dk_home_pct is None else round(float(dk_home_pct), 4),
                "dk_home_ml": dk.get("dk_home_ml"),
                "dk_away_ml": dk.get("dk_away_ml"),
                "spread_gap_model_minus_dk": None if dk_spread is None else round(model_spread - float(dk_spread), 2),
                "odds_source": None if not dk else ESPN_DK_BOOK,
                "odds_note": None
                if dk
                else "ESPN DK game lines missing or OFF; fail closed, no invented prices",
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": LIVE_MODEL_VERSION,
        "slate": {
            "label": f"ESPN regular-season week {args.week}",
            "start_date": args.start_date,
            "end_date": args.end_date,
            "games": len(games_out),
            "byes": list(BYES),
        },
        "published": False,
        "output_path": str(args.output),
        "sources": {
            "model": "frozen models/nfl_v2_live.joblib",
            "features": "nflverse schedules + PBP via build_nfl_feature_store (leakage-safe pregame)",
            "qb_overrides": "src.data.nfl_injury_news ESPN injuries + depth charts, Sleeper GSIS ids",
            "odds": "site.api.espn.com scoreboard competitions[].odds tagged espn_dk",
        },
        "credentials": {
            "bigquery": False,
            "supabase": False,
            "note": "No serving credentials in this environment; JSON cache only. Daily Refresh publishes to /markets/nfl when GCP/Supabase secrets are present.",
        },
        "news_summary": news_summary,
        "sleeper_error": sleeper_error,
        "posted_starter_overrides": len(posted),
        "games": games_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "games": len(games_out), "qb_overrides": len(posted)}, indent=2))


if __name__ == "__main__":
    main()
