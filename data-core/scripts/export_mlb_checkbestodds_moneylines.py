#!/usr/bin/env python3
"""
Export free MLB moneylines from CheckBestOdds historical pages.

CheckBestOdds publishes decimal 1/2 prices and event timestamps. The displayed
date can be Europe-local for evening US games, so joins use team tuple plus
nearest official MLB `game_datetime`.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://checkbestodds.com/baseball-odds/historical-odds-usa-mlb"

TEAM_ALIASES = {
    "Arizona": "Arizona Diamondbacks",
    "Athletics": "Athletics",
    "Atlanta": "Atlanta Braves",
    "Baltimore": "Baltimore Orioles",
    "Boston": "Boston Red Sox",
    "Cincinnati": "Cincinnati Reds",
    "Cleveland": "Cleveland Guardians",
    "Colorado": "Colorado Rockies",
    "Cubs": "Chicago Cubs",
    "Detroit": "Detroit Tigers",
    "Houston": "Houston Astros",
    "Kansas City": "Kansas City Royals",
    "LA Angels": "Los Angeles Angels",
    "LA Dodgers": "Los Angeles Dodgers",
    "Miami": "Miami Marlins",
    "Milwaukee": "Milwaukee Brewers",
    "Minnesota": "Minnesota Twins",
    "NY Mets": "New York Mets",
    "NY Yankees": "New York Yankees",
    "Philadelphia": "Philadelphia Phillies",
    "Pittsburgh": "Pittsburgh Pirates",
    "San Diego": "San Diego Padres",
    "San Francisco": "San Francisco Giants",
    "Seattle": "Seattle Mariners",
    "St. Louis": "St. Louis Cardinals",
    "Tampa Bay": "Tampa Bay Rays",
    "Texas": "Texas Rangers",
    "Toronto": "Toronto Blue Jays",
    "Washington": "Washington Nationals",
    "White Sox": "Chicago White Sox",
}


def norm_team(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def decimal_to_american(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    decimal = float(value)
    if decimal <= 1:
        return None
    if decimal >= 2:
        return int(round((decimal - 1) * 100))
    return int(round(-100 / (decimal - 1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLB moneylines from CheckBestOdds.")
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    parser.add_argument(
        "--games-path",
        default="data-core/notebooks/cache/mlb_games_2021_2026.parquet",
    )
    parser.add_argument("--start-date", default="2025-03-01")
    parser.add_argument("--end-date", default="2026-05-21")
    parser.add_argument("--max-hours-delta", type=float, default=18.0)
    parser.add_argument(
        "--output",
        default="data-core/notebooks/cache/mlb_checkbestodds_moneylines_2025_2026.csv",
    )
    parser.add_argument(
        "--audit-output",
        default="data-core/notebooks/cache/mlb_checkbestodds_moneylines_2025_2026_audit.json",
    )
    return parser.parse_args()


def fetch_year(year: int) -> pd.DataFrame:
    url = f"{BASE_URL}/{year}"
    response = requests.get(url, timeout=(10, 90), headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows: list[dict[str, object]] = []
    displayed_date: str | None = None

    for tr in soup.find_all("tr"):
        date_tag = tr.find("b", class_="dBY")
        if date_tag:
            displayed_date = date_tag.get_text(strip=True)
            continue

        link = tr.find("a", href=re.compile(r"/baseball-odds/usa-mlb/"))
        time_tag = tr.find("span", class_=re.compile(r"\btime\b"))
        if not link or not time_tag or not time_tag.has_attr("ts"):
            continue

        matchup = " ".join(link.get_text(" ", strip=True).split())
        if " - " not in matchup:
            continue
        home_raw, away_raw = [part.strip() for part in matchup.split(" - ", 1)]

        prices: list[float] = []
        for price_tag in tr.find_all("b"):
            try:
                prices.append(float(price_tag.get_text(strip=True).replace(",", ".")))
            except ValueError:
                continue
        if len(prices) < 2:
            continue

        rows.append(
            {
                "source_year": year,
                "source_displayed_date": displayed_date,
                "source_match_ts": pd.to_datetime(int(time_tag["ts"]), unit="s", utc=True),
                "home_raw": home_raw,
                "away_raw": away_raw,
                "home_team": TEAM_ALIASES.get(home_raw),
                "away_team": TEAM_ALIASES.get(away_raw),
                "home_decimal": prices[0],
                "away_decimal": prices[1],
                "source_href": link.get("href"),
            }
        )

    return pd.DataFrame(rows)


def load_games(path: str, start_date: str, end_date: str) -> pd.DataFrame:
    games = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.normalize()
    games["game_datetime"] = pd.to_datetime(games["game_datetime"], utc=True)
    games = games[(games["game_date"] >= pd.Timestamp(start_date)) & (games["game_date"] <= pd.Timestamp(end_date))]
    if "game_type" in games.columns:
        games = games[games["game_type"] == "R"]
    games = games.copy()
    games["home_key"] = games["home_team"].map(norm_team)
    games["away_key"] = games["away_team"].map(norm_team)
    return games


def join_odds_to_games(games: pd.DataFrame, odds: pd.DataFrame, max_hours_delta: float) -> pd.DataFrame:
    odds = odds.dropna(subset=["home_team", "away_team"]).copy()
    odds["home_key"] = odds["home_team"].map(norm_team)
    odds["away_key"] = odds["away_team"].map(norm_team)

    candidates = games.merge(
        odds,
        on=["home_key", "away_key"],
        how="left",
        suffixes=("", "_odds"),
    )
    candidates["hours_delta"] = (
        candidates["game_datetime"] - candidates["source_match_ts"]
    ).abs().dt.total_seconds() / 3600.0
    candidates = candidates[candidates["hours_delta"] <= max_hours_delta].copy()
    candidates = candidates.sort_values(["game_pk", "hours_delta", "source_match_ts"])
    joined = candidates.drop_duplicates(subset=["game_pk"], keep="first").copy()

    joined["home_moneyline"] = joined["home_decimal"].map(decimal_to_american)
    joined["away_moneyline"] = joined["away_decimal"].map(decimal_to_american)
    return joined[
        [
            "game_pk",
            "game_date",
            "home_team",
            "away_team",
            "home_moneyline",
            "away_moneyline",
            "home_decimal",
            "away_decimal",
            "source_match_ts",
            "hours_delta",
            "source_href",
        ]
    ].sort_values(["game_date", "game_pk"])


def summarize_missing(games: pd.DataFrame, joined: pd.DataFrame) -> list[dict[str, object]]:
    missing = games[~games["game_pk"].isin(set(joined["game_pk"]))].copy()
    return (
        missing[["game_pk", "game_date", "home_team", "away_team"]]
        .head(25)
        .assign(game_date=lambda df: df["game_date"].dt.date.astype(str))
        .to_dict(orient="records")
    )


def main() -> None:
    args = parse_args()

    odds_frames: list[pd.DataFrame] = []
    for year in args.years:
        odds_frames.append(fetch_year(year))
        time.sleep(1)
    odds = pd.concat(odds_frames, ignore_index=True)

    games = load_games(args.games_path, args.start_date, args.end_date)
    joined = join_odds_to_games(games, odds, args.max_hours_delta)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output, index=False)

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": BASE_URL,
        "years": args.years,
        "games_path": args.games_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "source_rows": int(len(odds)),
        "source_null_team_rows": int(odds[["home_team", "away_team"]].isna().any(axis=1).sum()),
        "target_games": int(len(games)),
        "matched_rows": int(len(joined)),
        "missing_rows": int(len(games) - len(joined)),
        "coverage_pct": float(len(joined) / len(games)) if len(games) else 0.0,
        "output": args.output,
        "missing_sample": summarize_missing(games, joined),
        "notes": [
            "Source is a public HTML odds comparison page, not an official API.",
            "Prices are decimal odds converted to American moneylines.",
            "Book is unspecified consensus/best odds, not Pinnacle.",
            "Rows join by home/away team and nearest official MLB game_datetime.",
        ],
    }
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved {len(joined)} MLB moneyline rows to {output}")
    print(f"Coverage: {len(joined)}/{len(games)} ({audit['coverage_pct']:.1%})")
    print(f"Saved audit to {audit_output}")


if __name__ == "__main__":
    main()
