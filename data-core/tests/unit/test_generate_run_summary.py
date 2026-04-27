import json

from scripts.generate_run_summary import (
    build_discord_payload,
    build_summary,
    load_validation_report,
    render_markdown,
)


def test_load_validation_report_missing_path(tmp_path):
    report, warning = load_validation_report(tmp_path / "missing.json")

    assert report is None
    assert "not found" in warning


def test_build_summary_warns_on_validation_failures():
    summary = build_summary(
        mode="normal",
        status="success",
        run_url="https://github.com/example/actions/runs/1",
        repository="dmboynton56/sports-edge",
        workflow="Daily Sports-Edge Refresh",
        validation={
            "recent_predictions": 12,
            "recent_games": 8,
            "recent_final_scores": 7,
            "final_missing_scores": 1,
            "orphan_predictions": 2,
        },
        validation_warning=None,
        snapshot={
            "prediction_hours": 48,
            "score_lookback_days": 7,
            "odds_hours": 24,
            "predictions_by_league": {"NBA": 9, "NFL": 3},
            "games_by_league": {"NBA": 8},
            "final_scores_by_league": {"NBA": 7},
            "odds_by_league": {
                "NBA": {
                    "count": 18,
                    "latest_snapshot_ts": "2026-04-27T13:00:00+00:00",
                }
            },
            "latest_odds_snapshot_ts": "2026-04-27T13:00:00+00:00",
        },
        query_error=None,
        generated_at="2026-04-27T13:05:00+00:00",
    )

    assert "1 final games are missing scores." in summary["warnings"]
    assert "2 orphan predictions were reported." in summary["warnings"]


def test_render_markdown_includes_validation_and_snapshot():
    summary = build_summary(
        mode="dry-run",
        status="success",
        run_url="https://github.com/example/actions/runs/2",
        repository="dmboynton56/sports-edge",
        workflow="Daily Sports-Edge Refresh",
        validation=None,
        validation_warning="Validation JSON was not found.",
        snapshot={
            "prediction_hours": 48,
            "score_lookback_days": 7,
            "odds_hours": 24,
            "predictions_by_league": {"NBA": 4},
            "games_by_league": {"NBA": 2},
            "final_scores_by_league": {"NBA": 1},
            "odds_by_league": {},
            "latest_odds_snapshot_ts": None,
        },
        query_error=None,
        generated_at="2026-04-27T13:05:00+00:00",
    )

    markdown = render_markdown(summary)

    assert "Sports Edge Refresh Summary" in markdown
    assert "Mode: `dry-run`" in markdown
    assert "Predictions by league: NBA: 4" in markdown
    assert "Dry run: Supabase write and strict validation steps were skipped." in markdown


def test_build_discord_payload_is_webhook_json():
    summary = build_summary(
        mode="normal",
        status="success",
        run_url="https://github.com/example/actions/runs/3",
        repository="dmboynton56/sports-edge",
        workflow="Daily Sports-Edge Refresh",
        validation={
            "recent_predictions": 3,
            "recent_games": 2,
            "recent_final_scores": 2,
            "final_missing_scores": 0,
            "orphan_predictions": 0,
        },
        validation_warning=None,
        snapshot=None,
        query_error="connection refused",
        generated_at="2026-04-27T13:05:00+00:00",
    )

    payload = build_discord_payload(summary)

    json.dumps(payload)
    assert payload["content"] == "**Sports-Edge Refresh succeeded**"
    assert payload["embeds"][0]["title"] == "Daily refresh success"
    assert payload["embeds"][0]["fields"][3]["name"] == "Validation"
    assert "connection refused" in payload["embeds"][0]["fields"][-1]["value"]
