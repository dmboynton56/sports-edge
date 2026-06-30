from __future__ import annotations

import pandas as pd

from src.models.mlb_hr_statcast_features import fetch_statcast


def test_fetch_statcast_uses_cache_when_it_covers_window(tmp_path, monkeypatch):
    cache = tmp_path / "mlb_statcast_2026.csv"
    pd.DataFrame({"game_date": ["2026-03-18", "2026-06-29"], "batter": [1, 2]}).to_csv(cache, index=False)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("fresh cache should not fetch Statcast chunks")

    monkeypatch.setattr("src.models.mlb_hr_statcast_features._fetch_statcast_chunk", fail_fetch)

    frame = fetch_statcast(
        pd.Timestamp("2026-03-02"),
        pd.Timestamp("2026-06-29"),
        cache=cache,
        refresh=False,
        sleep=0,
    )

    assert len(frame) == 2


def test_fetch_statcast_refreshes_stale_cache(tmp_path, monkeypatch):
    cache = tmp_path / "mlb_statcast_2026.csv"
    pd.DataFrame({"game_date": ["2026-03-18", "2026-06-15"], "batter": [1, 2]}).to_csv(cache, index=False)
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch_chunk(start, end, *, timeout):
        calls.append((start, end))
        return pd.DataFrame(
            {
                "game_date": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
                "batter": [10, 11],
            }
        )

    monkeypatch.setattr("src.models.mlb_hr_statcast_features._fetch_statcast_chunk", fetch_chunk)

    frame = fetch_statcast(
        pd.Timestamp("2026-03-02"),
        pd.Timestamp("2026-06-29"),
        cache=cache,
        refresh=False,
        chunk_days=30,
        sleep=0,
    )

    assert calls
    assert pd.to_datetime(frame["game_date"]).max() == pd.Timestamp("2026-06-29")
    assert pd.to_datetime(pd.read_csv(cache)["game_date"]).max() == pd.Timestamp("2026-06-29")
