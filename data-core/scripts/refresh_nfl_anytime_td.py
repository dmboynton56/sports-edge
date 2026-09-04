#!/usr/bin/env python3
"""Publish calibrated NFL anytime-TD predictions and current sportsbook odds."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from dotenv import load_dotenv
import nflreadpy as nfl
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fantasy.sleeper import (  # noqa: E402
    load_nflverse_rosters,
    load_sleeper_players,
    sleeper_availability,
)
from src.models.nfl_anytime_td import (  # noqa: E402
    AnytimeTDModel,
    FEATURE_COLUMNS,
    TARGET_POSITIONS,
    build_feature_frame,
)
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials  # noqa: E402
from src.utils.team_codes import canonical_nfl_abbr  # noqa: E402


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "americanfootball_nfl"
ODDS_MARKET = "player_anytime_td"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh NFL anytime touchdown market.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None, help="Defaults to the next scheduled week.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "nfl_anytime_td_v1.txt")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=ROOT / "models" / "nfl_anytime_td_v1_metrics.json",
    )
    parser.add_argument("--history-seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--minimum-history-games", type=int, default=3)
    parser.add_argument(
        "--near-kickoff-hours",
        type=int,
        default=72,
        help="Refresh existing prices only inside this many hours of kickoff.",
    )
    parser.add_argument("--minimum-refresh-hours", type=int, default=6)
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_player_name(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"(^|[ .])(jr|sr|ii|iii|iv|v)\.?($|[ .])", " ", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def american_implied_probability(price: float) -> float:
    return 100.0 / (price + 100.0) if price > 0 else abs(price) / (abs(price) + 100.0)


def _to_pandas(value):
    if hasattr(value, "collect"):
        value = value.collect()
    return value.to_pandas() if hasattr(value, "to_pandas") else value


def fetch_games(conn, *, season: int, week: int) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              g.id::text AS game_id,
              g.season,
              g.week,
              COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date) AS game_date,
              g.game_time_utc,
              g.home_team,
              g.away_team,
              totals.point AS total_line,
              td_odds.snapshot_ts AS td_odds_ts
            FROM games g
            LEFT JOIN LATERAL (
              SELECT o.line AS point
              FROM odds_snapshots o
              WHERE o.game_id = g.id AND o.market = 'total' AND o.selection = 'over'
              ORDER BY o.snapshot_ts DESC
              LIMIT 1
            ) totals ON true
            LEFT JOIN LATERAL (
              SELECT o.snapshot_ts
              FROM nfl_anytime_td_odds_snapshots o
              WHERE o.game_id = g.id
              ORDER BY o.snapshot_ts DESC
              LIMIT 1
            ) td_odds ON true
            WHERE g.league = 'NFL' AND g.season = %s AND g.week = %s
            ORDER BY g.game_time_utc, g.id
            """,
            (season, week),
            prepare=False,
        )
        columns = [description[0] for description in cur.description]
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def resolve_week(conn, *, season: int, requested_week: int | None) -> int:
    if requested_week is not None:
        return requested_week
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT week
            FROM games
            WHERE league = 'NFL'
              AND season = %s
              AND week IS NOT NULL
              AND game_time_utc >= NOW() - INTERVAL '3 hours'
            ORDER BY game_time_utc, week
            LIMIT 1
            """,
            (season,),
            prepare=False,
        )
        row = cur.fetchone()
    if not row:
        raise SystemExit(f"No upcoming Supabase NFL week found for {season}")
    return int(row[0])


def build_future_player_rows(
    games: list[dict[str, Any]],
    official_rosters: Mapping[str, Mapping[str, Any]],
    sleeper_players: Mapping[str, Mapping[str, Any]],
    *,
    season: int,
    week: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    sleeper_by_gsis = {
        str(player.get("gsis_id")): player
        for player in sleeper_players.values()
        if player.get("gsis_id")
    }
    sleeper_by_name_position = {
        (normalize_player_name(player.get("full_name")), str(player.get("position") or "").upper()): player
        for player in sleeper_players.values()
        if player.get("full_name") and player.get("position")
    }
    players_by_team: dict[str, list[dict[str, Any]]] = {}
    flags: dict[str, list[str]] = {}
    for player_id, player in official_rosters.items():
        if str(player.get("status") or "").upper() != "ACT":
            continue
        position = str(player.get("position") or "").upper()
        if position not in TARGET_POSITIONS:
            continue
        team = canonical_nfl_abbr(player.get("team"))
        if not team:
            continue
        player_name = str(player.get("full_name") or "").strip() or " ".join(
            str(player.get(field) or "").strip() for field in ("first_name", "last_name")
        ).strip()
        sleeper = sleeper_by_gsis.get(str(player_id))
        if sleeper is None:
            sleeper = sleeper_by_name_position.get((normalize_player_name(player_name), position))
        sleeper = sleeper or {}
        availability = sleeper_availability(sleeper)
        if availability in {"out", "inactive", "doubtful"}:
            continue
        if not player_name:
            continue
        quality_flags = []
        if not sleeper:
            quality_flags.append("roster_role_unverified")
        if availability == "questionable":
            quality_flags.append("questionable")
        depth_order = sleeper.get("depth_chart_order")
        if depth_order not in (None, ""):
            try:
                if int(depth_order) >= 3:
                    quality_flags.append("secondary_depth_role")
                if int(depth_order) >= 4:
                    quality_flags.append("deep_depth_chart")
            except (TypeError, ValueError):
                pass
        flags[str(player_id)] = quality_flags
        players_by_team.setdefault(team, []).append(
            {
                "player_id": str(player_id),
                "player_display_name": player_name,
                "position": position,
            }
        )

    rows: list[dict[str, Any]] = []
    for game in games:
        for scheduled_team, opponent in (
            (game["home_team"], game["away_team"]),
            (game["away_team"], game["home_team"]),
        ):
            roster_key = canonical_nfl_abbr(scheduled_team)
            for player in players_by_team.get(str(roster_key), []):
                rows.append(
                    {
                        **player,
                        "season": season,
                        "week": week,
                        "season_type": "REG",
                        "game_id": str(game["game_id"]),
                        "team": scheduled_team,
                        "opponent_team": opponent,
                        "is_future": True,
                    }
                )
    return pd.DataFrame(rows), flags


def build_prediction_rows(
    model: AnytimeTDModel,
    features: pd.DataFrame,
    games: list[dict[str, Any]],
    flags: Mapping[str, list[str]],
    *,
    season: int,
    week: int,
    minimum_history_games: int,
    prediction_ts: datetime,
) -> list[dict[str, Any]]:
    current = features[features["is_future"]].copy()
    current = current[current["career_games_before"] >= minimum_history_games].copy()
    current["td_probability"] = model.predict_proba(current)
    game_lookup = {str(game["game_id"]): game for game in games}
    rows: list[dict[str, Any]] = []
    for _, player in current.iterrows():
        game = game_lookup[str(player["game_id"])]
        player_id = str(player["player_id"])
        quality_flags = list(flags.get(player_id, []))
        if int(player["career_games_before"]) < 10:
            quality_flags.append("limited_history")
        if pd.isna(player.get("total_line")):
            quality_flags.append("missing_game_total")
        rows.append(
            {
                "game_id": str(player["game_id"]),
                "season": season,
                "week": week,
                "game_date": game["game_date"],
                "player_id": player_id,
                "player_name": str(player["player_display_name"]),
                "normalized_player_name": normalize_player_name(player["player_display_name"]),
                "team": str(player["team"]),
                "opponent": str(player["opponent_team"]),
                "position": str(player["position"]),
                "td_probability": float(player["td_probability"]),
                "sample_games": int(player["career_games_before"]),
                "model_version": model.model_version,
                "prediction_ts": prediction_ts,
                "quality_flags": quality_flags,
                "feature_snapshot": {
                    column: None if pd.isna(player.get(column)) else float(player[column])
                    for column in FEATURE_COLUMNS
                },
            }
        )
    return rows


def sync_predictions(conn, rows: list[dict[str, Any]], *, game_ids: list[str], model_version: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM nfl_anytime_td_predictions
            WHERE model_version = %s AND game_id = ANY(%s::uuid[])
            """,
            (model_version, game_ids),
            prepare=False,
        )
        for row in rows:
            cur.execute(
                """
                INSERT INTO nfl_anytime_td_predictions (
                  game_id, season, week, game_date, player_id, player_name,
                  normalized_player_name, team, opponent, position, td_probability,
                  sample_games, model_version, prediction_ts, quality_flags, feature_snapshot
                ) VALUES (
                  %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s::jsonb, %s::jsonb
                )
                """,
                (
                    row["game_id"], row["season"], row["week"], row["game_date"],
                    row["player_id"], row["player_name"], row["normalized_player_name"],
                    row["team"], row["opponent"], row["position"], row["td_probability"],
                    row["sample_games"], row["model_version"], row["prediction_ts"],
                    json.dumps(row["quality_flags"]), json.dumps(row["feature_snapshot"]),
                ),
                prepare=False,
            )
    conn.commit()
    return len(rows)


def _should_refresh_game(
    game: Mapping[str, Any],
    *,
    now: datetime,
    near_kickoff_hours: int,
    minimum_refresh_hours: int,
) -> bool:
    last_snapshot = game.get("td_odds_ts")
    if last_snapshot is None:
        return True
    if last_snapshot.tzinfo is None:
        last_snapshot = last_snapshot.replace(tzinfo=timezone.utc)
    if now - last_snapshot < timedelta(hours=minimum_refresh_hours):
        return False
    kickoff = game["game_time_utc"]
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    return kickoff <= now + timedelta(hours=near_kickoff_hours)


def _match_event(game: Mapping[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    game_home = canonical_nfl_abbr(game.get("home_team"))
    game_away = canonical_nfl_abbr(game.get("away_team"))
    kickoff = pd.to_datetime(game.get("game_time_utc"), utc=True, errors="coerce")
    candidates = []
    for event in events:
        if canonical_nfl_abbr(event.get("home_team")) != game_home:
            continue
        if canonical_nfl_abbr(event.get("away_team")) != game_away:
            continue
        event_time = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(kickoff) or pd.isna(event_time):
            continue
        delta = abs((event_time - kickoff).total_seconds())
        if delta <= 12 * 3600:
            candidates.append((delta, event))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def fetch_odds_rows(
    games: list[dict[str, Any]],
    *,
    api_key: str,
    now: datetime,
    near_kickoff_hours: int,
    minimum_refresh_hours: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "sports-edge/1.0"})
    response = session.get(
        f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/events",
        params={"apiKey": api_key, "dateFormat": "iso"},
        timeout=30,
    )
    response.raise_for_status()
    events = response.json()
    rows: list[dict[str, Any]] = []
    refreshed_games = 0
    skipped_cached = 0
    unmatched_games: list[str] = []
    remaining = response.headers.get("x-requests-remaining")

    for game in games:
        if not _should_refresh_game(
            game,
            now=now,
            near_kickoff_hours=near_kickoff_hours,
            minimum_refresh_hours=minimum_refresh_hours,
        ):
            skipped_cached += 1
            continue
        event = _match_event(game, events)
        if event is None:
            unmatched_games.append(f"{game['away_team']}@{game['home_team']}")
            continue
        odds_response = session.get(
            f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/events/{event['id']}/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": ODDS_MARKET,
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=30,
        )
        odds_response.raise_for_status()
        remaining = odds_response.headers.get("x-requests-remaining", remaining)
        refreshed_games += 1
        payload = odds_response.json()
        for bookmaker in payload.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != ODDS_MARKET:
                    continue
                for outcome in market.get("outcomes", []):
                    if str(outcome.get("name") or "").lower() != "yes":
                        continue
                    player_name = str(outcome.get("description") or "").strip()
                    price = float(outcome["price"])
                    if not player_name or price == 0:
                        continue
                    rows.append(
                        {
                            "game_id": str(game["game_id"]),
                            "provider_event_id": str(event["id"]),
                            "player_name": player_name,
                            "normalized_player_name": normalize_player_name(player_name),
                            "book": str(bookmaker.get("key") or "unknown"),
                            "book_title": bookmaker.get("title"),
                            "price": price,
                            "implied_probability": american_implied_probability(price),
                            "last_update": market.get("last_update") or bookmaker.get("last_update"),
                            "snapshot_ts": now,
                            "raw_record": outcome,
                        }
                    )
    return rows, {
        "refreshed_games": refreshed_games,
        "cached_games_skipped": skipped_cached,
        "unmatched_games": unmatched_games,
        "requests_remaining": int(remaining) if remaining is not None else None,
    }


def sync_odds(conn, rows: list[dict[str, Any]]) -> int:
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO nfl_anytime_td_odds_snapshots (
                  game_id, provider_event_id, player_name, normalized_player_name,
                  market, book, book_title, price, implied_probability, last_update,
                  snapshot_ts, source, raw_record
                ) VALUES (
                  %s::uuid, %s, %s, %s, 'player_anytime_td', %s, %s, %s, %s,
                  %s, %s, 'the_odds_api', %s::jsonb
                )
                """,
                (
                    row["game_id"], row["provider_event_id"], row["player_name"],
                    row["normalized_player_name"], row["book"], row["book_title"],
                    row["price"], row["implied_probability"], row["last_update"],
                    row["snapshot_ts"], json.dumps(row["raw_record"]),
                ),
                prepare=False,
            )
    conn.commit()
    return len(rows)


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    model, metadata = AnytimeTDModel.load(args.model_path, args.metadata_path)
    if not metadata.get("supportable"):
        raise SystemExit("Anytime-TD artifact failed its holdout gate")

    creds = load_supabase_credentials()
    conn = create_pg_connection(
        creds["url"], creds["db_password"], creds.get("db_host"),
        creds["db_port"], creds["db_name"], creds["db_user"],
    )
    try:
        week = resolve_week(conn, season=args.season, requested_week=args.week)
        games = fetch_games(conn, season=args.season, week=week)
        if not games:
            raise SystemExit(f"No Supabase NFL games found for {args.season} Week {week}")

        stats = _to_pandas(nfl.load_player_stats(args.history_seasons, summary_level="week"))
        stats = stats[stats["season_type"].eq("REG")].copy()
        history_schedule = _to_pandas(nfl.load_schedules(args.history_seasons))
        history_schedule = history_schedule[history_schedule["game_type"].eq("REG")].copy()
        future_players, player_flags = build_future_player_rows(
            games,
            load_nflverse_rosters(args.season),
            load_sleeper_players(),
            season=args.season,
            week=week,
        )
        current_schedule = pd.DataFrame(games)
        combined_stats = pd.concat([stats, future_players], ignore_index=True, sort=False)
        combined_schedule = pd.concat(
            [history_schedule, current_schedule], ignore_index=True, sort=False
        )
        features = build_feature_frame(combined_stats, combined_schedule)
        now = datetime.now(timezone.utc)
        prediction_rows = build_prediction_rows(
            model,
            features,
            games,
            player_flags,
            season=args.season,
            week=week,
            minimum_history_games=args.minimum_history_games,
            prediction_ts=now,
        )

        odds_rows: list[dict[str, Any]] = []
        odds_summary: dict[str, Any] = {"skipped": True}
        if not args.skip_odds:
            api_key = os.getenv("ODDS_API_KEY")
            if not api_key:
                raise SystemExit("ODDS_API_KEY is required unless --skip-odds is used")
            odds_rows, odds_summary = fetch_odds_rows(
                games,
                api_key=api_key,
                now=now,
                near_kickoff_hours=args.near_kickoff_hours,
                minimum_refresh_hours=args.minimum_refresh_hours,
            )

        if not args.dry_run:
            game_ids = [str(game["game_id"]) for game in games]
            synced_predictions = sync_predictions(
                conn, prediction_rows, game_ids=game_ids, model_version=model.model_version
            )
            synced_odds = sync_odds(conn, odds_rows) if odds_rows else 0
        else:
            synced_predictions = 0
            synced_odds = 0
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "season": args.season,
                "week": week,
                "games": len(games),
                "prediction_rows": len(prediction_rows),
                "synced_predictions": synced_predictions,
                "odds_rows": len(odds_rows),
                "synced_odds": synced_odds,
                "holdout_brier": metadata["brier"],
                "holdout_auc": metadata["auc"],
                "odds": odds_summary,
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
