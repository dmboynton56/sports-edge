"""
Create rerunnable MLB notebooks backed by the pipeline scripts.

The notebooks stay light: production logic lives in scripts/modules, and cells
load generated caches/metrics or call the scripts explicitly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


NOTEBOOK_DIR = Path("data-core/notebooks")


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_notebook(cells), f, indent=1)
        f.write("\n")
    print(f"Wrote {path}")


def main() -> None:
    _write(
        NOTEBOOK_DIR / "mlb_data_audit.ipynb",
        [
            _markdown(
                """
                # MLB Data Audit

                Audit MLB raw schedules, optional boxscore enrichment, and feature-store coverage.
                Run the backfill/build cells first when caches are stale.
                """
            ),
            _code(
                """
                import json
                import pandas as pd
                from pathlib import Path

                games_path = Path("cache/mlb_games_2021_2026.parquet")
                feature_path = Path("cache/mlb_feature_store_2021_2026.parquet")
                audit_path = Path("cache/mlb_feature_store_2021_2026_audit.json")
                """
            ),
            _code(
                """
                # Optional refresh from the public MLB Stats API.
                # %run ../scripts/backfill_mlb_raw.py --start-season 2021 --end-season 2026 --games-cache cache/mlb_games_2021_2026.parquet --refresh-games
                # %run ../scripts/build_mlb_feature_store.py --games-cache cache/mlb_games_2021_2026.parquet --output cache/mlb_feature_store_2021_2026.parquet --audit-output cache/mlb_feature_store_2021_2026_audit.json
                """
            ),
            _code(
                """
                games = pd.read_parquet(games_path)
                features = pd.read_parquet(feature_path)
                audit = json.loads(audit_path.read_text())
                audit
                """
            ),
            _code(
                """
                pd.DataFrame({
                    "artifact": ["games", "features"],
                    "rows": [len(games), len(features)],
                    "min_date": [games.game_date.min(), features.game_date.min()],
                    "max_date": [games.game_date.max(), features.game_date.max()],
                    "seasons": [sorted(games.season.unique()), sorted(features.season.unique())],
                })
                """
            ),
            _code(
                """
                features[[
                    "home_probable_pitcher_id",
                    "away_probable_pitcher_id",
                    "venue_id",
                ]].isna().mean().rename("missing_rate").to_frame()
                """
            ),
        ],
    )

    _write(
        NOTEBOOK_DIR / "mlb_model_training_backtest.ipynb",
        [
            _markdown(
                """
                # MLB Model Training And Backtest

                Train candidate MLB winner models from the feature store, hold out 2026 YTD,
                and write predictions/metrics for the performance history hub.
                """
            ),
            _code(
                """
                import json
                import pandas as pd
                from pathlib import Path

                metrics_path = Path("cache/mlb_backtest_metrics_2026_ytd.json")
                preds_path = Path("cache/mlb_backtest_predictions_2026_ytd.csv")
                """
            ),
            _code(
                """
                # Rebuild metrics and predictions.
                # %run ../scripts/backtest_mlb_winners.py --features-path cache/mlb_feature_store_2021_2026.parquet --validation-season 2025 --test-season 2026 --predictions-output cache/mlb_backtest_predictions_2026_ytd.csv --metrics-output cache/mlb_backtest_metrics_2026_ytd.json
                """
            ),
            _code(
                """
                metrics = json.loads(metrics_path.read_text())
                metrics["selected_refit_test"]
                """
            ),
            _code(
                """
                rows = []
                for name, values in metrics["candidate_metrics"].items():
                    row = {"model": name}
                    row.update({f"validation_{k}": v for k, v in values["validation"].items()})
                    row.update({f"test_{k}": v for k, v in values["test"].items()})
                    rows.append(row)
                pd.DataFrame(rows)
                """
            ),
            _code(
                """
                preds = pd.read_csv(preds_path)
                preds.assign(correct=preds.pick_won.astype(bool)).groupby("pick_side").agg(
                    games=("game_pk", "count"),
                    accuracy=("correct", "mean"),
                    avg_home_prob=("home_win_prob", "mean"),
                )
                """
            ),
        ],
    )

    _write(
        NOTEBOOK_DIR / "mlb_moneyline_roi.ipynb",
        [
            _markdown(
                """
                # MLB Moneyline ROI

                Join historical sportsbook moneylines to MLB backtest predictions.
                This notebook intentionally reports no ROI until an odds file is supplied.
                """
            ),
            _code(
                """
                import json
                import pandas as pd
                from pathlib import Path

                preds_path = Path("cache/mlb_backtest_predictions_2026_ytd.csv")
                metrics_path = Path("cache/mlb_backtest_metrics_2026_ytd.json")
                odds_path = Path("cache/mlb_moneylines_2025.csv")
                historical_audit_path = Path("cache/mlb_moneylines_historical_audit.json")
                """
            ),
            _code(
                """
                preds = pd.read_csv(preds_path)
                metrics = json.loads(metrics_path.read_text())
                metrics["odds_summary"]
                """
            ),
            _code(
                """
                # Probe/backfill from The Odds API historical endpoint.
                # Requires paid historical access; free keys write an audit with status=historical_unavailable.
                # %run ../scripts/backfill_historical_moneylines.py --sport MLB --games-path cache/mlb_games_2021_2026.parquet --start-date 2025-09-28 --end-date 2025-09-28 --limit-dates 1

                if historical_audit_path.exists():
                    json.loads(historical_audit_path.read_text())
                """
            ),
            _code(
                """
                if odds_path.exists():
                    odds = pd.read_csv(odds_path)
                    display(odds.head())
                else:
                    print(f"Missing {odds_path}; run backtest_mlb_winners.py with --odds-path after historical odds are restored.")
                """
            ),
        ],
    )

    _write(
        NOTEBOOK_DIR / "sports_model_performance_hub.ipynb",
        [
            _markdown(
                """
                # Sports Model Performance Hub

                Lightweight index of generated performance artifacts for NBA, NFL, MLB, PGA, and CBB.
                The source of truth remains the scripts plus docs/analysis markdown files.
                """
            ),
            _code(
                """
                import json
                from pathlib import Path
                import pandas as pd

                docs = Path("../docs/analysis")
                cache = Path("cache")
                """
            ),
            _code(
                """
                analysis_files = sorted(docs.glob("*_performance_*.md"))
                pd.DataFrame({"analysis_doc": [str(path) for path in analysis_files]})
                """
            ),
            _code(
                """
                mlb_metrics = cache / "mlb_backtest_metrics_2026_ytd.json"
                history_path = cache / "performance_history.json"
                if mlb_metrics.exists():
                    data = json.loads(mlb_metrics.read_text())
                    pd.DataFrame([{
                        "sport": "MLB",
                        "test_season": data["splits"]["test_season"],
                        **data["selected_refit_test"],
                        "odds_rows": data["odds_summary"]["odds_rows"],
                    }])
                """
            ),
            _code(
                """
                # Rebuild the consolidated history artifact.
                # %run ../scripts/export_performance_history.py

                if history_path.exists():
                    history = json.loads(history_path.read_text())
                    pd.json_normalize(history["sports"])
                """
            ),
        ],
    )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    main()
