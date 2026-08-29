"""Test that mid-tournament probabilities are properly merged into exported predictions."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_pga_tournament_dashboard import _merge_midtournament_into_predictions


def test_merge_replaces_pre_tournament_probabilities_with_midtournament_updates():
    """Mid-tournament win/top10/top20 probs should replace pre-tournament baseline."""
    predictions = [
        {
            "player": "Scottie Scheffler",
            "best_calibrated_target_win_prob": 0.13626,
            "best_calibrated_target_top10_prob": 0.45,
            "best_calibrated_target_top20_prob": 0.65,
            "best_calibrated_target_made_cut_prob": 0.92,
        },
        {
            "player": "Viktor Hovland",
            "best_calibrated_target_win_prob": 0.07696,
            "best_calibrated_target_top10_prob": 0.38,
            "best_calibrated_target_top20_prob": 0.58,
            "best_calibrated_target_made_cut_prob": 0.89,
        },
    ]

    midtournament = {
        "meta": {
            "rounds_completed": 2,
            "cut_after_round": 2,
            "tournament_key": "bmw_championship_2026",
        },
        "predictions": [
            {
                "player": "Scottie Scheffler",
                "pred_name": "Scottie Scheffler",
                "sim_win_pct": 24.5,
                "sim_top10_pct": 68.2,
                "sim_top20_pct": 82.1,
                "sim_make_cut_pct": 100.0,
                "current_pos": 1,
                "current_pos_display": "1",
                "to_par": -8.0,
                "to_par_display": "-8",
            },
            {
                "player": "Viktor Hovland",
                "pred_name": "Viktor Hovland",
                "sim_win_pct": 18.3,
                "sim_top10_pct": 62.5,
                "sim_top20_pct": 78.4,
                "sim_make_cut_pct": 100.0,
                "current_pos": 2,
                "current_pos_display": "2",
                "to_par": -7.0,
                "to_par_display": "-7",
            },
        ],
    }

    merged = _merge_midtournament_into_predictions(predictions, midtournament)

    assert len(merged) == 2
    
    # Scheffler: pre-tournament 13.6% win -> mid-tournament 24.5% win
    scheffler = next(p for p in merged if p["player"] == "Scottie Scheffler")
    assert scheffler["best_calibrated_target_win_prob"] == 0.245
    assert scheffler["best_calibrated_target_top10_prob"] == 0.682
    assert scheffler["best_calibrated_target_top20_prob"] == 0.821
    assert scheffler["best_calibrated_target_made_cut_prob"] == 1.0
    assert scheffler["pre_tournament_win_prob"] == 0.13626
    assert scheffler["rounds_completed"] == 2
    assert scheffler["current_position"] == 1
    assert scheffler["to_par"] == -8.0

    # Hovland: pre-tournament 7.7% win -> mid-tournament 18.3% win
    hovland = next(p for p in merged if p["player"] == "Viktor Hovland")
    assert hovland["best_calibrated_target_win_prob"] == 0.183
    assert hovland["best_calibrated_target_top10_prob"] == 0.625
    assert hovland["best_calibrated_target_top20_prob"] == 0.784
    assert hovland["pre_tournament_win_prob"] == 0.07696


def test_no_cut_event_sets_make_cut_to_100_percent():
    """No-cut events should force make_cut to 100% for all active players."""
    predictions = [
        {
            "player": "Ryan Gerard",
            "best_calibrated_target_win_prob": 0.00546,
            "best_calibrated_target_made_cut_prob": 0.45,
        }
    ]

    midtournament = {
        "meta": {
            "rounds_completed": 2,
            "cut_after_round": 999,  # No-cut indicator
            "tournament_key": "tour_championship_2026",
        },
        "predictions": [
            {
                "player": "Ryan Gerard",
                "pred_name": "Ryan Gerard",
                "sim_win_pct": 3.2,
                "sim_make_cut_pct": 100.0,
            }
        ],
    }

    merged = _merge_midtournament_into_predictions(predictions, midtournament)

    gerard = merged[0]
    assert gerard["best_calibrated_target_win_prob"] == 0.032
    assert gerard["best_calibrated_target_made_cut_prob"] == 1.0


def test_players_not_in_midtournament_keep_original_probs():
    """Players who missed cut (not in midtournament) should keep pre-tournament probs."""
    predictions = [
        {"player": "Made Cut", "best_calibrated_target_win_prob": 0.08},
        {"player": "Missed Cut", "best_calibrated_target_win_prob": 0.02},
    ]

    midtournament = {
        "meta": {"rounds_completed": 2, "cut_after_round": 2},
        "predictions": [
            {"player": "Made Cut", "pred_name": "Made Cut", "sim_win_pct": 12.0}
        ],
    }

    merged = _merge_midtournament_into_predictions(predictions, midtournament)

    made_cut = next(p for p in merged if p["player"] == "Made Cut")
    assert made_cut["best_calibrated_target_win_prob"] == 0.12

    missed_cut = next(p for p in merged if p["player"] == "Missed Cut")
    assert missed_cut["best_calibrated_target_win_prob"] == 0.02


def test_empty_midtournament_returns_predictions_unchanged():
    """If no mid-tournament data exists, return predictions as-is."""
    predictions = [{"player": "Test", "best_calibrated_target_win_prob": 0.05}]

    merged_none = _merge_midtournament_into_predictions(predictions, None)
    assert merged_none == predictions

    merged_empty = _merge_midtournament_into_predictions(predictions, {"predictions": []})
    assert merged_empty == predictions
