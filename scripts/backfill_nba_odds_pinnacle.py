#!/usr/bin/env python3
"""
Backfill BigQuery raw_nba_odds from Pinnacle/Kaggle nba_main_lines.csv format.

Example:
    python scripts/backfill_nba_odds_pinnacle.py --project learned-pier-478122-p7 --csv-path "../archive (6)/nba_main_lines.csv"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, date
from typing import Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

# Full team names (Pinnacle) -> abbreviations
PINNACLE_TEAM_TO_ABBREV = {
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


def team_name_to_abbrev(name: str) -> Optional[str]:
    """Convert Pinnacle full team name to abbreviation."""
    if pd.isna(name):
        return None
    key = str(name).strip().lower()
    return PINNACLE_TEAM_TO_ABBREV.get(key)


def schedule_team_to_odds_format(team: str) -> str:
    t = str(team).upper().strip()
    return SCHEDULE_TO_ODDS_MAP.get(t, t)


def transform_pinnacle_df(df: pd.DataFrame) -> pd.DataFrame:
    """Transform Pinnacle main_lines format to our raw_nba_odds schema."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["scrape_date"] = df["timestamp"].dt.date

    # Filter to known NBA teams (exclude All-Star, etc.)
    df["team1_abbrev"] = df["team1"].apply(team_name_to_abbrev)
    df["team2_abbrev"] = df["team2"].apply(team_name_to_abbrev)
    df = df[df["team1_abbrev"].notna() & df["team2_abbrev"].notna()]

    # Deduplicate: one row per (team1, team2, scrape_date) - take latest scrape
    df = df.sort_values("timestamp", ascending=True)
    df = df.drop_duplicates(subset=["team1_abbrev", "team2_abbrev", "scrape_date"], keep="last")

    # Use scrape_date as game_date for matching (NBA games are typically same calendar day)
    df["game_date"] = df["scrape_date"]

    rows = []
    utc_now = datetime.now(tz=timezone.utc)

    for _, row in df.iterrows():
        t1, t2 = row["team1_abbrev"], row["team2_abbrev"]
        game_date = row["game_date"]
        # Season: 2025-26 -> use 2025 (end year) for our schema
        season = 2025 if game_date.year >= 2025 else 2024
        if game_date.month >= 10:
            season = game_date.year
        else:
            season = game_date.year - 1

        raw_record = {}
        for k, v in row.items():
            if pd.isna(v):
                raw_record[k] = None
            elif isinstance(v, (pd.Timestamp, datetime, date)):
                raw_record[k] = v.isoformat() if hasattr(v, "isoformat") else str(v)
            elif hasattr(v, "item"):  # numpy scalar
                raw_record[k] = v.item()
            else:
                raw_record[k] = v

        # We don't know home/away yet - will resolve in match_game_ids
        # Use placeholder; match will overwrite with correct home/away
        home_team = t1  # placeholder
        away_team = t2
        game_id = f"NBA_{season}_{game_date}_{away_team}_{home_team}"

        # Spread: team1_spread is from team1's perspective. team2_spread = -team1_spread.
        # We'll store home spread after matching.
        spread_val = row.get("team1_spread")
        if pd.notna(spread_val) and str(spread_val).upper() != "N/A":
            try:
                spread_val = float(spread_val)
                price = None
                if pd.notna(row.get("team1_spread_odds")):
                    price = float(row["team1_spread_odds"])
                raw_json = json.dumps(raw_record)
                rows.append({
                    "game_id": game_id,
                    "league": "NBA",
                    "season": season,
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "team1_abbrev": t1,
                    "team2_abbrev": t2,
                    "team1_spread": spread_val,
                    "book": "pinnacle",
                    "market": "spread",
                    "line": spread_val,
                    "price": price,
                    "whos_favored": None,
                    "ingested_at": utc_now,
                    "raw_record": raw_json,
                })
            except (ValueError, TypeError):
                pass

        total_val = row.get("over_total")
        if pd.notna(total_val) and str(total_val) != "N/A":
            try:
                total_val = float(total_val)
                rows.append({
                    "game_id": game_id,
                    "league": "NBA",
                    "season": season,
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "team1_abbrev": t1,
                    "team2_abbrev": t2,
                    "team1_spread": None,
                    "book": "pinnacle",
                    "market": "total",
                    "line": total_val,
                    "price": None,
                    "whos_favored": None,
                    "ingested_at": utc_now,
                    "raw_record": json.dumps(raw_record),
                })
            except (ValueError, TypeError):
                pass

    return pd.DataFrame(rows)


def match_game_ids(client: bigquery.Client, project: str, odds_df: pd.DataFrame) -> pd.DataFrame:
    """Match odds to raw_schedules. Resolve home/away and convert spread to home perspective."""
    if odds_df.empty or "team1_abbrev" not in odds_df.columns:
        return odds_df

    print("Matching odds to games in raw_schedules...")
    date_min = odds_df["game_date"].min()
    date_max = odds_df["game_date"].max()

    query = f"""
        SELECT game_id, season, game_date, home_team, away_team
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE league = 'NBA'
          AND game_date BETWEEN @date_min AND @date_max
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("date_min", "DATE", date_min),
                          bigquery.ScalarQueryParameter("date_max", "DATE", date_max)]
    )
    schedule_df = client.query(query, job_config=job_config).to_dataframe()
    schedule_df["game_date"] = pd.to_datetime(schedule_df["game_date"]).dt.date

    print(f"  Found {len(schedule_df):,} games in raw_schedules.")

    # Build lookup: (date, team_a, team_b) -> (game_id, home_team, away_team)
    game_lookup = {}
    for _, s in schedule_df.iterrows():
        home = schedule_team_to_odds_format(s["home_team"])
        away = schedule_team_to_odds_format(s["away_team"])
        d = s["game_date"]
        game_lookup[(d, home, away)] = (s["game_id"], home, away)
        game_lookup[(d, away, home)] = (s["game_id"], home, away)

    matched = []
    for _, row in odds_df.iterrows():
        t1, t2 = row["team1_abbrev"], row["team2_abbrev"]
        d = row["game_date"]
        key1 = (d, t1, t2)
        key2 = (d, t2, t1)
        info = game_lookup.get(key1) or game_lookup.get(key2)
        line = row.get("line")
        if info:
            game_id, home_team, away_team = info
            if row["market"] == "spread" and line is not None:
                if t1 == home_team:
                    line = line
                else:
                    line = -line
        else:
            game_id = row["game_id"]
            home_team = t1
            away_team = t2

        matched.append({
            "game_id": game_id,
            "league": row["league"],
            "season": row["season"],
            "game_date": row["game_date"],
            "home_team": home_team,
            "away_team": away_team,
            "book": row["book"],
            "market": row["market"],
            "line": line,
            "price": row.get("price"),
            "whos_favored": row.get("whos_favored"),
            "ingested_at": row["ingested_at"],
            "raw_record": row["raw_record"],
        })

    out = pd.DataFrame(matched)
    n = (~out["game_id"].str.startswith("NBA_")).sum()
    print(f"Matched {n} odds rows to game_ids.")
    return out


def _load_dataframe(client: bigquery.Client, df: pd.DataFrame, table_id: str) -> None:
    if df.empty:
        return
    cols = ["game_id", "league", "season", "game_date", "home_team", "away_team", "book", "market", "line", "price", "whos_favored", "ingested_at", "raw_record"]
    df = df[[c for c in cols if c in df.columns]]
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
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    print(f"Loaded {len(df):,} rows into {table_id}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)
    print(f"Loading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df):,} rows.")

    odds_df = transform_pinnacle_df(df)
    print(f"Transformed to {len(odds_df):,} odds rows.")

    odds_df = match_game_ids(client, args.project, odds_df)

    if args.replace and not odds_df.empty:
        dmin, dmax = odds_df["game_date"].min(), odds_df["game_date"].max()
        print(f"Deleting existing pinnacle odds for {dmin} to {dmax}...")
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("dmin", "DATE", dmin),
                bigquery.ScalarQueryParameter("dmax", "DATE", dmax),
            ]
        )
        client.query(
            f"""
            DELETE FROM `{args.project}.sports_edge_raw.raw_nba_odds`
            WHERE league = 'NBA' AND book = 'pinnacle' AND game_date BETWEEN @dmin AND @dmax
            """,
            job_config=job_config,
        ).result()

    table_id = f"{args.project}.sports_edge_raw.raw_nba_odds"
    _load_dataframe(client, odds_df, table_id)
    print("Done.")


if __name__ == "__main__":
    main()
