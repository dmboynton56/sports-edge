from datetime import date, datetime, timezone

import pytest

from scripts.refresh_nfl_availability import build_context_rows


def test_build_context_rows_only_impacts_confirmed_official_absences():
    games = [
        {
            "game_id": "11111111-1111-1111-1111-111111111111",
            "game_date": date(2026, 9, 13),
            "home_team": "ARI",
            "away_team": "KC",
        }
    ]
    official = {
        "00-rb": {"full_name": "Reserve Runner", "team": "AZ", "position": "RB", "status": "RES"},
        "00-qb": {"full_name": "Active Quarterback", "team": "KC", "position": "QB", "status": "ACT"},
        "00-cut": {"full_name": "Released Receiver", "team": "KC", "position": "WR", "status": "CUT"},
        "00-ok": {"full_name": "Healthy Tight End", "team": "KC", "position": "TE", "status": "ACT"},
    }
    sleeper = {
        "1": {"gsis_id": "00-rb", "injury_status": "Out"},
        "2": {"gsis_id": "00-qb", "injury_status": "Questionable"},
        "3": {"gsis_id": "00-cut", "injury_status": "Out"},
        "4": {"gsis_id": "00-ok"},
    }
    history = {
        "00-rb": {"sample_size": 50, "player_value": 0.2, "usage_share": 0.25},
        "00-qb": {"sample_size": 300, "player_value": 0.3, "usage_share": 0.8},
    }

    rows, summary = build_context_rows(
        games,
        official,
        sleeper,
        history,
        season=2026,
        report_ts=datetime(2026, 9, 3, tzinfo=timezone.utc),
    )

    assert [row["player_name"] for row in rows] == ["Reserve Runner", "Active Quarterback"]
    reserve, questionable = rows
    assert reserve["team"] == "ARI"
    assert reserve["status"] == "injured_reserve"
    assert reserve["team_delta"] == pytest.approx(-0.05)
    assert reserve["metric_name"] == "epa_per_play"
    assert questionable["status"] == "questionable"
    assert "metric_name" not in questionable
    assert summary["availability_reports"] == 2
    assert summary["confirmed_unavailable"] == 1
    assert summary["impact_estimates"] == 1
    assert summary["inactive_skipped"] == 1


def test_build_context_rows_never_turns_negative_epa_absence_into_upgrade():
    rows, _ = build_context_rows(
        [
            {
                "game_id": "11111111-1111-1111-1111-111111111111",
                "game_date": date(2026, 9, 13),
                "home_team": "DEN",
                "away_team": "LV",
            }
        ],
        {
            "00-wr": {
                "full_name": "Negative EPA Receiver",
                "team": "DEN",
                "position": "WR",
                "status": "PUP",
            }
        },
        {},
        {"00-wr": {"sample_size": 80, "player_value": -0.1, "usage_share": 0.2}},
        season=2026,
        report_ts=datetime(2026, 9, 3, tzinfo=timezone.utc),
    )

    assert rows[0]["team_delta"] == 0.0
