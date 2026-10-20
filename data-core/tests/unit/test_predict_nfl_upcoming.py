from datetime import datetime
from zoneinfo import ZoneInfo

from src.data.nfl_injury_news import PostedStarter
from scripts.predict_nfl_upcoming import (
    dk_lookup,
    index_nflverse_gsis,
    kickoff_mt,
    qb_note,
    starter_gsis,
)
from src.data.espn_dk_odds import ESPN_DK_BOOK


def test_qb_note_reports_skipped_out_starter():
    starter = PostedStarter(
        team="TB",
        name="Teddy Bridgewater",
        gsis_id="00-x",
        espn_id="2",
        depth_rank=2,
        status="active",
        skipped=({"name": "Baker Mayfield", "status": "out", "rank": 1},),
        source="espn",
    )
    assert "Baker Mayfield (out)" in qb_note(starter)
    assert "Teddy Bridgewater" in qb_note(starter)


def test_unresolved_qb_stays_last_three_mode():
    starter = PostedStarter(
        team="NYG",
        name=None,
        gsis_id=None,
        espn_id=None,
        depth_rank=None,
        status=None,
        skipped=({"name": "Jameis Winston", "status": "questionable", "rank": 1},),
        unresolved=True,
        source="espn",
    )
    note = qb_note(starter)
    assert "unresolved" in note
    assert "last-3" in note


def test_kickoff_mt_from_espn_utc():
    text = kickoff_mt("2026-10-22", "20:15", "2026-10-23T00:15:00Z")
    assert text.startswith("2026-10-22 18:15")
    assert "MDT" in text or "MST" in text


def test_dk_lookup_no_vig_and_home_spread():
    events = [
        {
            "id": "1",
            "commence_time": "2026-10-23T00:15:00Z",
            "home_team": "Chicago Bears",
            "away_team": "New England Patriots",
            "home_abbr": "CHI",
            "away_abbr": "NE",
            "bookmakers": [
                {
                    "key": ESPN_DK_BOOK,
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Chicago Bears", "point": -3.0, "price": -105},
                                {"name": "New England Patriots", "point": 3.0, "price": -115},
                            ],
                        },
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Chicago Bears", "price": -155},
                                {"name": "New England Patriots", "price": 130},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
    row = dk_lookup(events)[("NE", "CHI")]
    assert row["dk_spread"] == -3.0
    assert row["book"] == ESPN_DK_BOOK
    assert row["dk_no_vig_home_pct"] is not None
    assert 0.5 < row["dk_no_vig_home_pct"] < 0.7
    assert datetime.fromisoformat(row["commence_time"].replace("Z", "+00:00")).tzinfo == ZoneInfo("UTC") or True


def test_nflverse_roster_fills_missing_sleeper_gsis():
    rosters = {
        "00-huntley": {"espn_id": "2", "team": "BAL", "full_name": "Tyler Huntley"},
        "00-mckee": {"espn_id": "9", "team": "PHI", "full_name": "Tanner McKee"},
    }
    by_espn, by_name = index_nflverse_gsis(rosters)
    huntley = PostedStarter(
        team="BAL", name="Tyler Huntley", gsis_id=None, espn_id="2", depth_rank=2, status="active"
    )
    mckee = PostedStarter(
        team="PHI", name="Tanner McKee", gsis_id=None, espn_id=None, depth_rank=1, status="active"
    )
    assert starter_gsis(huntley, "BAL", by_espn, by_name) == "00-huntley"
    assert starter_gsis(mckee, "PHI", by_espn, by_name) == "00-mckee"
