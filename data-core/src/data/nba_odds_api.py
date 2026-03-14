"""
Fetch NBA odds from The Odds API and load into BigQuery raw_nba_odds.

Requires ODDS_API_KEY in environment. Used by daily refresh and predict_nba_picks.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from google.cloud import bigquery

# Odds API team names (lowercase) -> our abbreviations
ODDS_API_TEAM_TO_ABBREV = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "los angeles clippers": "LAC",
    "los angeles lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}

SCHEDULE_TO_ODDS_MAP = {
    "GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS",
    "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "BRX": "BKN",
}


def _team_to_abbrev(name: str) -> Optional[str]:
    if pd.isna(name):
        return None
    key = str(name).strip().lower()
    return ODDS_API_TEAM_TO_ABBREV.get(key)


def _schedule_to_odds(team: str) -> str:
    t = str(team).upper().strip()
    return SCHEDULE_TO_ODDS_MAP.get(t, t)


def fetch_and_transform_odds(date_str: str) -> pd.DataFrame:
    """
    Fetch NBA odds from The Odds API for a date and transform to raw_nba_odds schema.

    Returns DataFrame with: game_id, league, season, game_date, home_team, away_team,
    book, market, line, price, whos_favored, ingested_at, raw_record.
    """
    from src.data.odds_fetcher import fetch_odds

    df = fetch_odds("nba", date=date_str, markets="spreads,totals")
    if df.empty:
        return df

    date_obj = pd.to_datetime(date_str).date()
    season = date_obj.year if date_obj.month >= 10 else date_obj.year - 1
    utc_now = datetime.now(tz=timezone.utc)

    rows = []
    for _, row in df.iterrows():
        home_abbrev = _team_to_abbrev(row["home_team"])
        away_abbrev = _team_to_abbrev(row["away_team"])
        if home_abbrev is None or away_abbrev is None:
            continue

        market = row["market"]
        if market == "spreads":
            # outcome: name (team), point (spread), price (American odds)
            # Output one row per game per book with home spread. Only emit when outcome is home team.
            outcome_name = str(row.get("outcome_name", "")).strip().lower()
            home_lower = str(row["home_team"]).strip().lower()
            if outcome_name != home_lower:
                continue  # Skip away outcome; we'll get home from the other row or use -point
            point = row.get("line")
            if pd.isna(point):
                continue
            home_spread = float(point)

            raw_record = {
                "api_game_id": row.get("game_id"),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "book": row["book"],
                "outcome_name": row.get("outcome_name"),
                "point": point,
                "price": row.get("price"),
            }
            rows.append({
                "game_id": row["game_id"],  # API id, will be replaced when matching
                "league": "NBA",
                "season": season,
                "game_date": date_obj,
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "book": row["book"],
                "market": "spread",
                "line": home_spread,
                "price": float(row["price"]) if pd.notna(row.get("price")) else None,
                "whos_favored": None,
                "ingested_at": utc_now,
                "raw_record": json.dumps(raw_record),
            })
        elif market == "totals":
            # totals: Over/Under with point (total). Use Over outcome only (same line as Under)
            if str(row.get("outcome_name", "")).strip().lower() != "over":
                continue
            point = row.get("line")
            if pd.notna(point):
                raw_record = {
                    "api_game_id": row.get("game_id"),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "book": row["book"],
                    "outcome_name": row.get("outcome_name"),
                    "point": point,
                    "price": row.get("price"),
                }
                rows.append({
                    "game_id": row["game_id"],
                    "league": "NBA",
                    "season": season,
                    "game_date": date_obj,
                    "home_team": home_abbrev,
                    "away_team": away_abbrev,
                    "book": row["book"],
                    "market": "total",
                    "line": float(point),
                    "price": float(row["price"]) if pd.notna(row.get("price")) else None,
                    "whos_favored": None,
                    "ingested_at": utc_now,
                    "raw_record": json.dumps(raw_record),
                })

    return pd.DataFrame(rows)


def match_to_schedule(client: bigquery.Client, project: str, odds_df: pd.DataFrame) -> pd.DataFrame:
    """Match Odds API rows to raw_schedules game_ids by (game_date, home_team, away_team)."""
    if odds_df.empty:
        return odds_df

    date_min = odds_df["game_date"].min()
    date_max = odds_df["game_date"].max()

    query = f"""
        SELECT game_id, game_date, home_team, away_team
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE league = 'NBA'
          AND game_date BETWEEN @date_min AND @date_max
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date_min", "DATE", date_min),
            bigquery.ScalarQueryParameter("date_max", "DATE", date_max),
        ]
    )
    schedule = client.query(query, job_config=job_config).to_dataframe()
    schedule["game_date"] = pd.to_datetime(schedule["game_date"]).dt.date

    # Build lookup: (date, home_odds_format, away_odds_format) -> game_id
    game_lookup = {}
    for _, s in schedule.iterrows():
        home = _schedule_to_odds(s["home_team"])
        away = _schedule_to_odds(s["away_team"])
        d = s["game_date"]
        game_lookup[(d, home, away)] = s["game_id"]
        game_lookup[(d, away, home)] = s["game_id"]

    matched = []
    for _, row in odds_df.iterrows():
        key1 = (row["game_date"], row["home_team"], row["away_team"])
        key2 = (row["game_date"], row["away_team"], row["home_team"])
        game_id = game_lookup.get(key1) or game_lookup.get(key2)
        if game_id is None:
            continue
        row = row.copy()
        row["game_id"] = game_id
        matched.append(row)

    return pd.DataFrame(matched)


def fetch_and_load_odds_for_range(
    project: str,
    start_date: str,
    end_date: str,
    replace_existing: bool = True,
) -> int:
    """
    Fetch NBA odds for each date in [start_date, end_date] and load into raw_nba_odds.

    Args:
        start_date, end_date: YYYY-MM-DD strings (inclusive).
    Returns:
        Total number of odds rows loaded.
    """
    import pandas as pd

    total = 0
    for d in pd.date_range(start=start_date, end=end_date):
        date_str = d.strftime("%Y-%m-%d")
        n = fetch_and_load_odds(project, date_str, replace_existing=replace_existing)
        total += n
    return total


def fetch_and_load_odds(
    project: str,
    date_str: str,
    replace_existing: bool = True,
) -> int:
    """
    Fetch NBA odds for a date from The Odds API and load into raw_nba_odds.

    Returns number of rows loaded.
    """
    client = bigquery.Client(project=project)

    odds_df = fetch_and_transform_odds(date_str)
    if odds_df.empty:
        return 0

    odds_df = match_to_schedule(client, project, odds_df)

    if odds_df.empty:
        return 0

    if replace_existing:
        date_obj = pd.to_datetime(date_str).date()
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("d", "DATE", date_obj),
            ]
        )
        client.query(
            f"""
            DELETE FROM `{project}.sports_edge_raw.raw_nba_odds`
            WHERE league = 'NBA' AND game_date = @d
            """,
            job_config=job_config,
        ).result()

    cols = ["game_id", "league", "season", "game_date", "home_team", "away_team", "book", "market", "line", "price", "whos_favored", "ingested_at", "raw_record"]
    load_df = odds_df[[c for c in cols if c in odds_df.columns]]

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("league", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("season", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("game_date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("home_team", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("away_team", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("book", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("market", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("line", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("price", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("whos_favored", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("raw_record", "STRING", mode="NULLABLE"),
        ],
    )
    client.load_table_from_dataframe(load_df, f"{project}.sports_edge_raw.raw_nba_odds", job_config=job_config).result()
    return len(load_df)
