import pandas as pd

from src.fantasy.projection_model import normalize_weekly_stats


def test_normalize_weekly_stats_excludes_postseason_rows():
    weekly = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_display_name": "Example Runner",
                "position": "RB",
                "team": "BUF",
                "season": 2025,
                "week": 17,
                "season_type": "REG",
                "rushing_yards": 100,
            },
            {
                "player_id": "p1",
                "player_display_name": "Example Runner",
                "position": "RB",
                "team": "BUF",
                "season": 2025,
                "week": 18,
                "season_type": "POST",
                "rushing_yards": 200,
            },
        ]
    )

    normalized = normalize_weekly_stats(weekly)

    assert normalized["week"].tolist() == [17]
    assert normalized["rushing_yards"].tolist() == [100.0]
