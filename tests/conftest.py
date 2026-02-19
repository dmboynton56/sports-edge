import pytest
import pandas as pd

@pytest.fixture
def mock_nba_game_logs():
    return pd.DataFrame({
        'team': ['BOS', 'BOS', 'LAL', 'LAL'],
        'game_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02']),
        'pts': [100, 110, 105, 115]
    })
