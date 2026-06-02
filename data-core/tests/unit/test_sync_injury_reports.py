from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts.sync_injury_reports_to_supabase import build_payloads


def test_build_payloads_creates_availability_and_derived_impact():
    rows = [
        {
            "league": "NBA",
            "game_date": "2026-01-02",
            "team": "BOS",
            "opponent": "NYK",
            "player_name": "J.Tatum",
            "player_id": "nba-1",
            "position": "F",
            "status": "Out",
            "report_ts": "2026-01-02T18:00:00Z",
            "metric_name": "net_rating",
            "player_value": 8.0,
            "replacement_value": -2.0,
            "usage_share": 0.32,
            "sample_size": 42,
        }
    ]

    availability, impacts = build_payloads(
        rows,
        default_source="test_source",
        default_model_version="injury-impact-v1",
    )

    assert len(availability) == 1
    assert availability[0].league == "NBA"
    assert availability[0].status == "out"
    assert availability[0].source == "test_source"
    assert availability[0].game_date.isoformat() == "2026-01-02"

    assert len(impacts) == 1
    assert impacts[0].metric_name == "net_rating"
    assert impacts[0].team_delta == -3.2
    assert impacts[0].model_version == "injury-impact-v1"
    assert impacts[0].sample_size == 42


def test_build_payloads_accepts_explicit_team_delta_and_defaults_timestamps():
    default_ts = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
    rows = [
        {
            "league": "NFL",
            "season": 2026,
            "game_date": "2026-09-10",
            "team": "DEN",
            "player_name": "B.Nix",
            "status": "Questionable",
            "source": "manual",
            "metric_name": "epa_per_play",
            "team_delta": -0.117,
            "model_version": "injury-qb-v1",
        }
    ]

    availability, impacts = build_payloads(
        rows,
        default_source="ignored",
        default_model_version="ignored",
        default_report_ts=default_ts,
    )

    assert availability[0].report_ts == default_ts
    assert availability[0].status == "questionable"
    assert impacts[0].estimated_at == default_ts
    assert impacts[0].team_delta == pytest.approx(-0.117)
    assert impacts[0].season == 2026
    assert impacts[0].model_version == "injury-qb-v1"


def test_build_payloads_skips_rows_without_required_identity():
    availability, impacts = build_payloads(
        [
            {"league": "NBA", "team": "BOS", "status": "out"},
            {"league": "NBA", "player_name": "J.Tatum", "status": "out"},
        ],
        default_source="test",
        default_model_version="v1",
    )

    assert availability == []
    assert impacts == []


def test_build_payloads_skips_impact_without_delta_inputs():
    availability, impacts = build_payloads(
        [
            {
                "league": "NFL",
                "team": "DEN",
                "player_name": "B.Nix",
                "metric_name": "epa_per_play",
                "player_value": 0.2,
            }
        ],
        default_source="test",
        default_model_version="v1",
    )

    assert availability == []
    assert impacts == []


def test_build_payloads_handles_pandas_csv_scalar_values():
    rows = pd.DataFrame(
        [
            {
                "league": "NFL",
                "season": 2026,
                "game_date": "2026-09-10",
                "team": "DEN",
                "player_name": "B.Nix",
                "status": "Out",
                "metric_name": "epa_per_play",
                "player_value": 0.20,
                "replacement_value": 0.02,
                "usage_share": 0.65,
                "sample_size": 100,
            }
        ]
    ).to_dict("records")

    availability, impacts = build_payloads(
        rows,
        default_source="csv",
        default_model_version="injury-impact-v1",
    )

    assert availability[0].raw_record["season"] == 2026
    assert impacts[0].team_delta == pytest.approx(-0.117)
    assert impacts[0].sample_size == 100
