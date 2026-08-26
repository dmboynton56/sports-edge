#!/usr/bin/env python3
"""
Audit current MLB odds market coverage from The Odds API.

Scoped to active MLB events for today and the next ~24h. Reports:
- Events sampled
- Percent with each market present
- Median book count per market
- Small player-prop identity-match sample
- API credits used

Quota-aware: samples current events only, no historical backfill. Caps requests.
Estimates cost before fetching.

If ODDS_API_KEY is missing, writes a report that says "skipped: missing key".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from json_utils import dumps_strict  # noqa: E402
from src.data.markets_registry import get_markets_by_sport, load_markets_registry  # noqa: E402
from src.data.mlb_fetcher import fetch_mlb_schedule  # noqa: E402
from src.data.mlb_hr_odds_fetcher import (  # noqa: E402
    SLATE_TIMEZONE,
    SPORT_KEY,
    OddsApiClient,
    fetch_event_hr_odds,
    fetch_mlb_events,
    match_events_to_schedule,
)

DEFAULT_CACHE_DIR = ROOT / "notebooks" / "cache"
DEFAULT_ANALYSIS_DIR = ROOT / "docs" / "analysis"
DEFAULT_REGIONS = "us"


def estimate_request_cost(num_events: int, num_markets: int) -> int:
    """Estimate The Odds API request cost.
    
    Featured markets (h2h, spreads, totals) cost 1 request per sport.
    Event-level markets (props) cost 1 request per event per unique market returned.
    
    Conservative estimate for event-level props: num_events * num_markets.
    """
    # This is a conservative upper bound; actual cost may be lower
    # since The Odds API charges per unique returned market, not requested market
    return num_events * num_markets


def fetch_featured_odds(
    client: OddsApiClient,
    *,
    sport_key: str = SPORT_KEY,
    regions: str = DEFAULT_REGIONS,
    markets: list[str],
) -> dict[str, Any]:
    """Fetch featured odds (h2h, spreads, totals) for a sport.
    
    These are cheaper: 1 request for all events for requested markets.
    """
    payload = client.get(
        f"/sports/{sport_key}/odds",
        {
            "regions": regions,
            "markets": ",".join(markets),
            "dateFormat": "iso",
            "oddsFormat": "american",
        },
    )
    return payload if isinstance(payload, list) else []


def audit_market_coverage(
    events: list[dict[str, Any]],
    event_map: dict[str, dict[str, Any]],
    markets: list[dict[str, Any]],
    client: OddsApiClient,
    *,
    regions: str = DEFAULT_REGIONS,
    max_prop_samples: int = 5,
) -> dict[str, Any]:
    """Audit coverage for registered MLB markets.
    
    Args:
        events: List of event dicts from The Odds API
        event_map: Mapping from provider event_id to MLB schedule metadata
        markets: List of market entries from markets.yaml
        client: OddsApiClient instance
        regions: Regions to request
        max_prop_samples: Maximum number of events to sample for prop markets
        
    Returns:
        Audit dict with coverage stats per market
    """
    results: dict[str, Any] = {}
    
    # Separate featured markets (sport-level) from props (event-level)
    featured_market_keys = {"h2h", "spreads", "totals"}
    featured_markets = [
        m for m in markets if m["provider_market_key"] in featured_market_keys
    ]
    prop_markets = [
        m for m in markets if m["provider_market_key"] not in featured_market_keys
    ]
    
    # Fetch featured markets (1 request for all events)
    if featured_markets:
        featured_keys = [m["provider_market_key"] for m in featured_markets]
        featured_odds = fetch_featured_odds(
            client,
            sport_key=SPORT_KEY,
            regions=regions,
            markets=featured_keys,
        )
        
        # Build event_id -> markets map
        featured_event_markets: dict[str, dict[str, list[Any]]] = {}
        for event_payload in featured_odds:
            event_id = str(event_payload.get("id") or "")
            featured_event_markets[event_id] = {}
            
            for bookmaker in event_payload.get("bookmakers") or []:
                for market in bookmaker.get("markets") or []:
                    market_key = market.get("key")
                    if market_key not in featured_event_markets[event_id]:
                        featured_event_markets[event_id][market_key] = []
                    featured_event_markets[event_id][market_key].append(market)
        
        # Compute coverage stats for featured markets
        for market_entry in featured_markets:
            market_key = market_entry["provider_market_key"]
            market_id = market_entry["market_id"]
            
            events_with_market = 0
            book_counts = []
            
            for event in events:
                event_id = str(event.get("id") or "")
                if event_id not in event_map:
                    continue
                
                markets_for_event = featured_event_markets.get(event_id, {})
                if market_key in markets_for_event:
                    events_with_market += 1
                    # Count unique bookmakers
                    # We'd need to track books per event, but for now estimate from first event
                    book_counts.append(len(markets_for_event[market_key]))
            
            matched_events = len([e for e in events if str(e.get("id") or "") in event_map])
            coverage_pct = (
                100.0 * events_with_market / matched_events if matched_events > 0 else 0.0
            )
            median_books = float(pd.Series(book_counts).median()) if book_counts else 0.0
            
            results[market_id] = {
                "market_id": market_id,
                "provider_market_key": market_key,
                "subject_type": market_entry["subject_type"],
                "events_sampled": matched_events,
                "events_with_market": events_with_market,
                "coverage_percent": round(coverage_pct, 2),
                "median_books": median_books,
                "identity_sample": None,  # N/A for team markets
            }
    
    # Sample prop markets (expensive: 1 request per event)
    if prop_markets:
        prop_keys = [m["provider_market_key"] for m in prop_markets]
        sampled_events = [e for e in events if str(e.get("id") or "") in event_map]
        sampled_events = sampled_events[:max_prop_samples]
        
        for market_entry in prop_markets:
            market_key = market_entry["provider_market_key"]
            market_id = market_entry["market_id"]
            
            events_with_market = 0
            book_counts = []
            identity_samples = []
            
            for event in sampled_events:
                event_id = str(event.get("id") or "")
                try:
                    payload = fetch_event_hr_odds(
                        client,
                        event_id=event_id,
                        sport_key=SPORT_KEY,
                        regions=regions,
                        markets=[market_key],
                    )
                except Exception as exc:
                    # Don't fail entire audit on one event error
                    continue
                
                has_market = False
                unique_books = set()
                unique_players = set()
                
                for bookmaker in payload.get("bookmakers") or []:
                    for market in bookmaker.get("markets") or []:
                        if market.get("key") == market_key:
                            has_market = True
                            unique_books.add(bookmaker.get("key"))
                            
                            for outcome in market.get("outcomes") or []:
                                player = (
                                    outcome.get("description")
                                    or outcome.get("participant")
                                    or outcome.get("player")
                                )
                                if player:
                                    unique_players.add(player)
                
                if has_market:
                    events_with_market += 1
                    book_counts.append(len(unique_books))
                    
                    if len(identity_samples) < 3 and unique_players:
                        meta = event_map[event_id]
                        identity_samples.append(
                            {
                                "game_id": meta.get("game_id"),
                                "away_team": meta.get("away_team"),
                                "home_team": meta.get("home_team"),
                                "players_found": sorted(unique_players)[:5],
                            }
                        )
            
            matched_events = len(sampled_events)
            coverage_pct = (
                100.0 * events_with_market / matched_events if matched_events > 0 else 0.0
            )
            median_books = float(pd.Series(book_counts).median()) if book_counts else 0.0
            
            results[market_id] = {
                "market_id": market_id,
                "provider_market_key": market_key,
                "subject_type": market_entry["subject_type"],
                "events_sampled": matched_events,
                "events_with_market": events_with_market,
                "coverage_percent": round(coverage_pct, 2),
                "median_books": median_books,
                "identity_sample": identity_samples if identity_samples else None,
            }
    
    return results


def write_analysis_markdown(
    audit: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a markdown analysis report."""
    lines = [
        f"# MLB Odds Market Coverage Audit",
        f"",
        f"Generated: {audit['generated_at']}",
        f"",
        f"## Summary",
        f"",
        f"- **Slate date range:** {audit['slate_date_range']}",
        f"- **MLB events discovered:** {audit['mlb_events_discovered']}",
        f"- **MLB events matched:** {audit['mlb_events_matched']}",
        f"- **Markets evaluated:** {audit['markets_evaluated']}",
        f"- **API requests:** {audit['api_requests']}",
        f"- **Credits used:** {audit['credits_used']}",
        f"- **Credits remaining:** {audit['credits_remaining']}",
        f"",
    ]
    
    if audit.get("skipped"):
        lines.extend(
            [
                f"## Status",
                f"",
                f"**SKIPPED:** {audit['skipped']}",
                f"",
            ]
        )
    else:
        lines.extend(
            [
                f"## Coverage by Market",
                f"",
                f"| Market | Provider Key | Type | Events Sampled | Coverage % | Median Books |",
                f"| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        
        for market_id, data in audit.get("market_coverage", {}).items():
            lines.append(
                f"| {market_id} | {data['provider_market_key']} | "
                f"{data['subject_type']} | {data['events_sampled']} | "
                f"{data['coverage_percent']:.1f}% | {data['median_books']:.1f} |"
            )
        
        lines.extend(
            [
                f"",
                f"## Player Identity Sample",
                f"",
            ]
        )
        
        for market_id, data in audit.get("market_coverage", {}).items():
            if data.get("identity_sample"):
                lines.append(f"### {market_id}")
                lines.append("")
                for sample in data["identity_sample"][:2]:
                    lines.append(f"- **{sample['game_id']}:** {sample['away_team']} @ {sample['home_team']}")
                    lines.append(f"  - Players: {', '.join(sample['players_found'])}")
                lines.append("")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MLB odds market coverage from The Odds API"
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=datetime.now(ZoneInfo(SLATE_TIMEZONE)).date(),
        help="Start date for slate (default: today in America/Denver)",
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="End date for slate (default: start_date + 1 day)",
    )
    parser.add_argument(
        "--max-prop-samples",
        type=int,
        default=5,
        help="Maximum events to sample for prop markets (default: 5)",
    )
    parser.add_argument(
        "--regions",
        default=DEFAULT_REGIONS,
        help="Odds API regions (default: us)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: notebooks/cache/odds_market_coverage_YYYY-MM-DD.json)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path (default: docs/analysis/odds_market_coverage_YYYY-MM-DD.md)",
    )
    parser.add_argument(
        "--allow-missing-key",
        action="store_true",
        help="Write a 'skipped' report if ODDS_API_KEY is missing instead of failing",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(ROOT / ".env", override=False)
    args = parse_args()
    
    start_date = args.start_date
    end_date = args.end_date or start_date
    
    date_str = start_date.isoformat()
    output_json = args.output_json or (
        DEFAULT_CACHE_DIR / f"odds_market_coverage_{date_str}.json"
    )
    output_md = args.output_md or (
        DEFAULT_ANALYSIS_DIR / f"odds_market_coverage_{date_str}.md"
    )
    
    # Load markets registry
    try:
        registry = load_markets_registry()
        mlb_markets = get_markets_by_sport("MLB", registry=registry)
    except Exception as exc:
        print(f"Failed to load markets registry: {exc}")
        sys.exit(1)
    
    # Check for API key
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        if not args.allow_missing_key:
            print("ODDS_API_KEY not found in environment")
            sys.exit(1)
        
        # Write skipped report
        audit = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "slate_date_range": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "mlb_events_discovered": 0,
            "mlb_events_matched": 0,
            "markets_evaluated": len(mlb_markets),
            "api_requests": 0,
            "credits_used": None,
            "credits_remaining": None,
            "skipped": "ODDS_API_KEY not found in environment",
            "market_coverage": {},
        }
        
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(dumps_strict(audit, indent=2), encoding="utf-8")
        
        output_md.parent.mkdir(parents=True, exist_ok=True)
        write_analysis_markdown(audit, output_md)
        
        print(f"Wrote skipped audit to {output_json}")
        print(f"Wrote skipped analysis to {output_md}")
        return
    
    # Fetch MLB schedule
    season = start_date.year
    schedule = fetch_mlb_schedule(
        season,
        start_date=start_date,
        end_date=end_date,
        include_uncompleted=True,
    )
    
    # Fetch events from The Odds API
    client = OddsApiClient(api_key=api_key)
    
    # For simplicity, fetch events for start_date
    # (The Odds API doesn't have a date range param, so we'd need multiple calls)
    events = fetch_mlb_events(client, game_date=start_date)
    event_map = match_events_to_schedule(events, schedule)
    
    # Estimate cost before proceeding
    featured_market_count = sum(
        1 for m in mlb_markets if m["provider_market_key"] in {"h2h", "spreads", "totals"}
    )
    prop_market_count = len(mlb_markets) - featured_market_count
    
    estimated_cost = 1 + (min(len(events), args.max_prop_samples) * prop_market_count)
    print(f"Estimated API cost: {estimated_cost} credits")
    print(f"  - Featured markets: 1 credit")
    print(f"  - Prop markets: {min(len(events), args.max_prop_samples)} events × {prop_market_count} markets")
    print()
    
    # Audit coverage
    market_coverage = audit_market_coverage(
        events,
        event_map,
        mlb_markets,
        client,
        regions=args.regions,
        max_prop_samples=args.max_prop_samples,
    )
    
    # Build audit report
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "slate_date_range": f"{start_date.isoformat()} to {end_date.isoformat()}",
        "mlb_events_discovered": len(events),
        "mlb_events_matched": len(event_map),
        "markets_evaluated": len(mlb_markets),
        "api_requests": client.request_count,
        "credits_used": client.response_meta.requests_used,
        "credits_remaining": client.response_meta.requests_remaining,
        "market_coverage": market_coverage,
    }
    
    # Write outputs
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(dumps_strict(audit, indent=2), encoding="utf-8")
    
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_analysis_markdown(audit, output_md)
    
    print(f"Wrote coverage audit to {output_json}")
    print(f"Wrote analysis report to {output_md}")
    print()
    print(f"API requests: {client.request_count}")
    print(f"Credits used: {client.response_meta.requests_used}")
    print(f"Credits remaining: {client.response_meta.requests_remaining}")


if __name__ == "__main__":
    main()
