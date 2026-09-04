from src.fantasy.sleeper import merge_sleeper_context
from scripts.refresh_fantasy_context import propagate_weekly_context


def test_sleeper_context_updates_team_and_injury_by_gsis_id():
    rows = [
        {
            "player_id": "00-001",
            "player_name": "Example Runner",
            "position": "RB",
            "team": "OLD",
            "availability": "expected",
            "explanation": [],
        }
    ]
    summary = merge_sleeper_context(
        rows,
        {
            "123": {
                "gsis_id": "00-001",
                "full_name": "Example Runner",
                "position": "RB",
                "team": "BUF",
                "status": "Active",
                "injury_status": "Questionable",
                "injury_body_part": "Hamstring",
                "depth_chart_order": 1,
                "news_updated": 1_788_000_000_000,
            }
        },
    )

    assert summary == {"matched": 1, "unmatched": 0, "questionable": 1, "unavailable": 0}
    assert rows[0]["team"] == "BUF"
    assert rows[0]["availability"] == "questionable"
    assert rows[0]["depth_chart_order"] == 1
    assert "Hamstring" in rows[0]["explanation"][-1]


def test_sleeper_context_marks_ir_and_inactive_players_unavailable():
    rows = [
        {"player_id": "missing", "player_name": "Player One", "position": "WR", "team": "NYJ"},
        {"player_id": "missing-2", "player_name": "Player Two", "position": "TE", "team": "DAL"},
    ]
    summary = merge_sleeper_context(
        rows,
        {
            "1": {"full_name": "Player One", "position": "WR", "team": "NYJ", "status": "Active", "injury_status": "IR"},
            "2": {"full_name": "Player Two", "position": "TE", "team": "DAL", "status": "Inactive"},
        },
    )

    assert [row["availability"] for row in rows] == ["out", "inactive"]
    assert summary["unavailable"] == 2


def test_sleeper_context_does_not_guess_ambiguous_name_matches():
    rows = [{"player_id": "missing", "player_name": "Same Name", "position": "WR", "team": "SEA"}]
    summary = merge_sleeper_context(
        rows,
        {
            "1": {"full_name": "Same Name", "position": "WR", "team": "BUF", "status": "Active"},
            "2": {"full_name": "Same Name", "position": "WR", "team": "DAL", "status": "Active"},
        },
    )

    assert summary["matched"] == 0
    assert summary["unmatched"] == 1
    assert "availability" not in rows[0]


def test_official_roster_status_overrides_stale_sleeper_membership():
    rows = [{"player_id": "00-001", "player_name": "Example Runner", "position": "RB", "team": "OLD"}]
    summary = merge_sleeper_context(
        rows,
        {
            "1": {
                "gsis_id": "00-001",
                "full_name": "Example Runner",
                "position": "RB",
                "team": "OLD",
                "status": "Active",
            }
        },
        {"00-001": {"team": "BUF", "status": "DEV"}},
    )

    assert rows[0]["team"] == "BUF"
    assert rows[0]["official_roster_status"] == "DEV"
    assert rows[0]["availability"] == "inactive"
    assert summary["official_matched"] == 1
    assert summary["official_active"] == 0


def test_weekly_rows_inherit_refreshed_preseason_availability():
    payload = {
        "projections": [
            {"player_id": "one", "team": "BUF", "availability": "questionable", "injury_status": "Questionable"}
        ],
        "weekly": {"1": [{"player_id": "one", "availability": "expected"}]},
    }

    assert propagate_weekly_context(payload) == 1
    assert payload["weekly"]["1"][0]["availability"] == "questionable"
    assert payload["weekly"]["1"][0]["injury_status"] == "Questionable"
