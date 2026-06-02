from scripts.sync_evaluation_history_to_supabase import build_evaluation_payloads


def test_build_evaluation_payloads_creates_model_and_strategy_rows():
    history = {
        "sports": [
            {
                "sport": "NBA",
                "model_version": "v3",
                "season": "2025-26",
                "market": "spread",
                "data_source": "BigQuery backtest + Supabase ATS",
                "odds_status": "partial_historical_spread_odds",
                "sample": {
                    "completed_games": 1175,
                    "supabase_graded_games": 64,
                },
                "metrics": {
                    "bigquery_default_bets": 591,
                    "bigquery_default_wins": 301,
                    "bigquery_default_accuracy": 0.509,
                    "bigquery_default_roi": -0.005,
                    "bigquery_best_sweep_bets": 112,
                    "bigquery_best_sweep_edge_threshold": 0.5,
                    "bigquery_best_sweep_min_confidence": 0.4,
                    "best_reported_sweep_roi": 0.083,
                    "supabase_ats_wins": 31,
                    "supabase_ats_losses": 33,
                    "supabase_ats_pushes": 0,
                    "supabase_ats_roi": -0.075,
                    "bigquery_brier": 0.24,
                },
                "artifact_refs": ["scripts/backtest_nba_spread.py"],
                "gaps": ["odds gap"],
            },
            {
                "sport": "MLB",
                "model_version": "v3",
                "season": "2026 YTD",
                "market": "moneyline",
                "data_source": "MLB Stats API",
                "odds_status": "free_moneyline_history",
                "sample": {"odds_joined_games": 673},
                "metrics": {
                    "brier": 0.247,
                    "log_loss": 0.688,
                    "roc_auc": 0.543,
                    "ece_10": 0.012,
                    "flat_roi": -0.031,
                },
                "artifact_refs": ["scripts/backtest_mlb_winners.py"],
                "gaps": [],
            },
        ]
    }

    evaluations, strategies = build_evaluation_payloads(history)

    assert [row.league for row in evaluations] == ["NBA", "MLB"]
    assert evaluations[0].evaluation_name == "performance_history_nba_2025-26_spread"
    assert evaluations[0].model_name == "sports_edge"
    assert evaluations[0].metrics["sample"]["completed_games"] == 1175
    assert evaluations[0].calibration["bigquery_brier"] == 0.24
    assert evaluations[1].calibration == {
        "brier": 0.247,
        "log_loss": 0.688,
        "roc_auc": 0.543,
        "ece_10": 0.012,
    }

    strategy_ids = [row.strategy_id for row in strategies]
    assert strategy_ids == [
        "bigquery_default_edge_1_conf_0",
        "bigquery_best_reported_sweep",
        "supabase_ats_flat_minus_110",
        "moneyline_flat_pick",
    ]
    nba_default = strategies[0]
    assert nba_default.league == "NBA"
    assert nba_default.bets == 591
    assert nba_default.wins == 301
    assert nba_default.roi == -0.005
    assert nba_default.edge_threshold == 1.0
    assert nba_default.min_confidence == 0.0

    mlb_strategy = strategies[-1]
    assert mlb_strategy.league == "MLB"
    assert mlb_strategy.bets == 673
    assert mlb_strategy.roi == -0.031


def test_build_evaluation_payloads_skips_strategy_when_no_roi_metric():
    history = {
        "sports": [
            {
                "sport": "CBB",
                "model_version": "manual matchup artifacts",
                "season": "CV 2016-2025",
                "market": "tournament winner probability",
                "data_source": "cache",
                "odds_status": "no_sportsbook_odds",
                "sample": {"folds": 9},
                "metrics": {"log_loss": 0.575, "brier": 0.198, "auc": 0.758},
                "artifact_refs": [],
                "gaps": [],
            }
        ]
    }

    evaluations, strategies = build_evaluation_payloads(history)

    assert len(evaluations) == 1
    assert evaluations[0].league == "CBB"
    assert evaluations[0].calibration == {
        "log_loss": 0.575,
        "brier": 0.198,
        "auc": 0.758,
    }
    assert strategies == []
