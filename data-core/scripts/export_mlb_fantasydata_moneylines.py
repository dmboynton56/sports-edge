#!/usr/bin/env python3
"""
Export free MLB consensus moneylines from public FantasyData team odds pages.

This is mainly useful for 2026 YTD, where CheckBestOdds starts later than the
local MLB schedule cache. Rows join by date, teams, and final score to handle
same-day doubleheaders.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://fantasydata.com"

FANTASYDATA_TO_MLB_STATS_ABBR = {
    "ARI": "AZ",
    "ATH": "ATH",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CHW": "CWS",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KC",
    "LAA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYY": "NYY",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SD",
    "SEA": "SEA",
    "SF": "SF",
    "STL": "STL",
    "TB": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WSH",
}


def parse_number(value: str) -> float | None:
    text = str(value or "").strip().replace("+", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_price(value: str) -> int | None:
    number = parse_number(value)
    return int(number) if number is not None else None


def team_abbr_from_cell(cell) -> str | None:
    span = cell.select_one("span.md-show")
    if not span:
        return None
    return FANTASYDATA_TO_MLB_STATS_ABBR.get(span.get_text(strip=True).upper())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLB moneylines from FantasyData public pages.")
    parser.add_argument(
        "--games-path",
        default="data-core/notebooks/cache/mlb_games_2021_2026.parquet",
    )
    parser.add_argument("--start-date", default="2026-03-01")
    parser.add_argument("--end-date", default="2026-05-21")
    parser.add_argument(
        "--output",
        default="data-core/notebooks/cache/mlb_fantasydata_moneylines_2026_ytd.csv",
    )
    parser.add_argument(
        "--audit-output",
        default="data-core/notebooks/cache/mlb_fantasydata_moneylines_2026_ytd_audit.json",
    )
    return parser.parse_args()


def discover_team_links() -> list[str]:
    response = requests.get(f"{BASE_URL}/mlb/odds", timeout=(10, 30), headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = {
        link.get("href")
        for link in soup.find_all("a", href=re.compile(r"^/mlb/.+-odds$"))
        if link.get("href")
    }
    return sorted(links)


def fetch_team_page(path: str) -> pd.DataFrame:
    url = f"{BASE_URL}{path}"
    response = requests.get(url, timeout=(10, 30), headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    rows: list[dict[str, object]] = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 15:
            continue

        away_abbr = team_abbr_from_cell(cells[0])
        home_abbr = team_abbr_from_cell(cells[1])
        game_date = pd.to_datetime(cells[2].get_text(strip=True), errors="coerce")
        if not away_abbr or not home_abbr or pd.isna(game_date):
            continue

        away_moneyline = parse_price(cells[10].get_text(strip=True))
        home_moneyline = parse_price(cells[11].get_text(strip=True))
        if away_moneyline is None or home_moneyline is None:
            continue

        rows.append(
            {
                "game_date": game_date.normalize(),
                "away_team_abbr": away_abbr,
                "home_team_abbr": home_abbr,
                "away_score": parse_price(cells[3].get_text(strip=True)),
                "home_score": parse_price(cells[4].get_text(strip=True)),
                "home_moneyline": home_moneyline,
                "away_moneyline": away_moneyline,
                "source_url": url,
            }
        )

    return pd.DataFrame(rows)


def fetch_all_pages() -> pd.DataFrame:
    frames = []
    for path in discover_team_links():
        frames.append(fetch_team_page(path))
        time.sleep(0.2)
    odds = pd.concat(frames, ignore_index=True)
    odds = odds.drop_duplicates(
        subset=["game_date", "home_team_abbr", "away_team_abbr", "home_score", "away_score"],
        keep="first",
    )
    return odds.sort_values(["game_date", "home_team_abbr", "away_team_abbr"]).reset_index(drop=True)


def load_games(path: str, start_date: str, end_date: str) -> pd.DataFrame:
    games = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.normalize()
    games = games[(games["game_date"] >= pd.Timestamp(start_date)) & (games["game_date"] <= pd.Timestamp(end_date))]
    if "game_type" in games.columns:
        games = games[games["game_type"] == "R"]
    return games.copy()


def build_output(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    merged = games.merge(
        odds,
        on=["game_date", "home_team_abbr", "away_team_abbr", "home_score", "away_score"],
        how="inner",
    )
    return merged[
        [
            "game_pk",
            "game_date",
            "home_team",
            "away_team",
            "home_moneyline",
            "away_moneyline",
            "source_url",
        ]
    ].sort_values(["game_date", "game_pk"])


def main() -> None:
    args = parse_args()
    odds = fetch_all_pages()
    games = load_games(args.games_path, args.start_date, args.end_date)
    out = build_output(games, odds)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    missing = games[~games["game_pk"].isin(set(out["game_pk"]))]
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://fantasydata.com/mlb/{team}-odds",
        "games_path": args.games_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "source_rows_deduped": int(len(odds)),
        "target_games": int(len(games)),
        "matched_rows": int(len(out)),
        "missing_rows": int(len(missing)),
        "coverage_pct": float(len(out) / len(games)) if len(games) else 0.0,
        "output": args.output,
        "missing_sample": (
            missing[["game_pk", "game_date", "home_team", "away_team"]]
            .head(25)
            .assign(game_date=lambda df: df["game_date"].dt.date.astype(str))
            .to_dict(orient="records")
        ),
        "notes": [
            "Source is public HTML, not an official API.",
            "Book is FantasyData consensus, not Pinnacle.",
            "Rows join by date, teams, and final score to avoid doubleheader ambiguity.",
        ],
    }
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved {len(out)} MLB moneyline rows to {output}")
    print(f"Coverage: {len(out)}/{len(games)} ({audit['coverage_pct']:.1%})")
    print(f"Saved audit to {audit_output}")


if __name__ == "__main__":
    main()
