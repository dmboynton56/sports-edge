#!/usr/bin/env python3
"""
Export free NBA consensus spreads from public FantasyData team odds pages.

The output matches the `raw_nba_odds` CSV shape used by the repo's NBA backtests:
`game_id`, `book`, `market=spread`, `line`, and `price`, with `line` from the
home team's perspective.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.team_codes import NBA_TEAM_NAME_TO_ABBR, canonical_nba_abbr


BASE_URL = "https://fantasydata.com"


TEAM_SLUGS = {
    "atlanta hawks": "atlanta-hawks",
    "boston celtics": "boston-celtics",
    "brooklyn nets": "brooklyn-nets",
    "charlotte hornets": "charlotte-hornets",
    "chicago bulls": "chicago-bulls",
    "cleveland cavaliers": "cleveland-cavaliers",
    "dallas mavericks": "dallas-mavericks",
    "denver nuggets": "denver-nuggets",
    "detroit pistons": "detroit-pistons",
    "golden state warriors": "golden-state-warriors",
    "houston rockets": "houston-rockets",
    "indiana pacers": "indiana-pacers",
    "la clippers": "los-angeles-clippers",
    "la lakers": "los-angeles-lakers",
    "memphis grizzlies": "memphis-grizzlies",
    "miami heat": "miami-heat",
    "milwaukee bucks": "milwaukee-bucks",
    "minnesota timberwolves": "minnesota-timberwolves",
    "new orleans pelicans": "new-orleans-pelicans",
    "new york knicks": "new-york-knicks",
    "oklahoma city thunder": "oklahoma-city-thunder",
    "orlando magic": "orlando-magic",
    "philadelphia 76ers": "philadelphia-76ers",
    "phoenix suns": "phoenix-suns",
    "portland trail blazers": "portland-trail-blazers",
    "sacramento kings": "sacramento-kings",
    "san antonio spurs": "san-antonio-spurs",
    "toronto raptors": "toronto-raptors",
    "utah jazz": "utah-jazz",
    "washington wizards": "washington-wizards",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NBA spreads from FantasyData public pages.")
    parser.add_argument(
        "--games-path",
        default="data-core/notebooks/cache/nba_backtest_2025_v3.csv",
    )
    parser.add_argument("--start-date", default="2026-02-13")
    parser.add_argument("--end-date", default="2026-05-21")
    parser.add_argument(
        "--output",
        default="data-core/notebooks/cache/nba_fantasydata_spreads_2026_tail.csv",
    )
    parser.add_argument(
        "--audit-output",
        default="data-core/notebooks/cache/nba_fantasydata_spreads_2026_tail_audit.json",
    )
    return parser.parse_args()


def team_abbr_from_cell(cell) -> str | None:
    span = cell.select_one("span.md-show")
    if span:
        return canonical_nba_abbr(span.get_text(strip=True))
    link = cell.find("a")
    if link:
        return canonical_nba_abbr(link.get_text(" ", strip=True))
    return canonical_nba_abbr(cell.get_text(" ", strip=True))


def fetch_team_page(slug: str) -> pd.DataFrame:
    url = f"{BASE_URL}/nba/{slug}-odds"
    response = requests.get(url, timeout=(10, 30), headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    rows: list[dict[str, object]] = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 15:
            continue

        away = team_abbr_from_cell(cells[0])
        home = team_abbr_from_cell(cells[1])
        game_date = pd.to_datetime(cells[2].get_text(strip=True), errors="coerce")
        if not away or not home or pd.isna(game_date):
            continue

        home_spread = parse_number(cells[7].get_text(strip=True))
        home_price = parse_price(cells[8].get_text(strip=True))
        if home_spread is None or home_price is None:
            continue

        rows.append(
            {
                "game_date": game_date.normalize(),
                "home_team": home,
                "away_team": away,
                "home_spread": home_spread,
                "home_price": home_price,
                "away_spread": parse_number(cells[5].get_text(strip=True)),
                "away_price": parse_price(cells[6].get_text(strip=True)),
                "source_url": url,
            }
        )

    return pd.DataFrame(rows)


def fetch_all_pages() -> pd.DataFrame:
    frames = []
    for slug in sorted(set(TEAM_SLUGS.values())):
        frames.append(fetch_team_page(slug))
        time.sleep(0.2)
    odds = pd.concat(frames, ignore_index=True)
    odds = odds.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")
    return odds.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)


def load_target_games(path: str, start_date: str, end_date: str) -> pd.DataFrame:
    games = pd.read_csv(path)
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.normalize()
    games = games[(games["game_date"] >= pd.Timestamp(start_date)) & (games["game_date"] <= pd.Timestamp(end_date))]
    games = games.copy()
    games["home_team"] = games["home_team"].map(lambda value: canonical_nba_abbr(value) or str(value).upper())
    games["away_team"] = games["away_team"].map(lambda value: canonical_nba_abbr(value) or str(value).upper())
    return games


def build_output(target_games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    merged = target_games.merge(odds, on=["game_date", "home_team", "away_team"], how="inner")
    utc_now = datetime.now(timezone.utc)
    out = pd.DataFrame(
        {
            "game_id": merged["game_id"].astype(str),
            "league": "NBA",
            "season": 2025,
            "game_date": merged["game_date"].dt.date,
            "home_team": merged["home_team"],
            "away_team": merged["away_team"],
            "book": "fantasydata_consensus",
            "market": "spread",
            "line": merged["home_spread"].astype(float),
            "price": merged["home_price"].astype(int),
            "whos_favored": merged["home_spread"].map(lambda value: "home" if float(value) < 0 else "away"),
            "ingested_at": utc_now,
            "raw_record": merged.apply(
                lambda row: json.dumps(
                    {
                        "source": "fantasydata_public_team_odds",
                        "source_url": row["source_url"],
                        "away_spread": row["away_spread"],
                        "away_price": row["away_price"],
                    }
                ),
                axis=1,
            ),
        }
    )
    return out.sort_values(["game_date", "game_id"])


def main() -> None:
    args = parse_args()
    odds = fetch_all_pages()
    target_games = load_target_games(args.games_path, args.start_date, args.end_date)
    out = build_output(target_games, odds)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    missing = target_games[~target_games["game_id"].astype(str).isin(set(out["game_id"].astype(str)))]
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://fantasydata.com/nba/{team}-odds",
        "games_path": args.games_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "source_rows_deduped": int(len(odds)),
        "target_games": int(len(target_games)),
        "matched_rows": int(len(out)),
        "missing_rows": int(len(missing)),
        "coverage_pct": float(len(out) / len(target_games)) if len(target_games) else 0.0,
        "output": args.output,
        "missing_sample": (
            missing[["game_id", "game_date", "home_team", "away_team"]]
            .head(25)
            .assign(game_date=lambda df: df["game_date"].dt.date.astype(str))
            .to_dict(orient="records")
        ),
        "notes": [
            "Source is public HTML, not an official API.",
            "Book is FantasyData consensus, not Pinnacle.",
            "Line and price are home-perspective spread values.",
        ],
    }
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved {len(out)} NBA spread rows to {output}")
    print(f"Coverage: {len(out)}/{len(target_games)} ({audit['coverage_pct']:.1%})")
    print(f"Saved audit to {audit_output}")


if __name__ == "__main__":
    main()
