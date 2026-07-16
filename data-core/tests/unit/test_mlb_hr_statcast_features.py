from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import requests

from src.models.mlb_hr_statcast_features import (
    _fetch_statcast_chunk,
    fetch_statcast,
    preload_statcast_cache,
)


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


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


def test_fetch_statcast_chunk_retries_retryable_http(monkeypatch):
    responses = [
        _Response(502),
        _Response(200, "game_date,batter,pitcher\n2026-04-21,10,20\n"),
    ]
    calls = []

    def get(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr("src.models.mlb_hr_statcast_features.requests.get", get)
    monkeypatch.setattr("src.models.mlb_hr_statcast_features.time.sleep", lambda *_args, **_kwargs: None)

    frame = _fetch_statcast_chunk(
        pd.Timestamp("2026-04-21"),
        pd.Timestamp("2026-04-27"),
        timeout=1,
        max_attempts=2,
        retry_sleep=0,
    )

    assert len(calls) == 2
    assert frame.loc[0, "batter"] == 10


def test_fetch_statcast_uses_stale_cache_when_chunks_fail(tmp_path, monkeypatch):
    cache = tmp_path / "mlb_statcast_2026.csv"
    pd.DataFrame({"game_date": ["2026-03-18", "2026-06-15"], "batter": [1, 2]}).to_csv(cache, index=False)

    def fetch_chunk(start, end, *, timeout):
        raise requests.HTTPError("502 Server Error")

    monkeypatch.setattr("src.models.mlb_hr_statcast_features._fetch_statcast_chunk", fetch_chunk)

    with pytest.warns(RuntimeWarning, match="partial/stale cache"):
        frame = fetch_statcast(
            pd.Timestamp("2026-03-02"),
            pd.Timestamp("2026-06-29"),
            cache=cache,
            refresh=False,
            chunk_days=30,
            sleep=0,
        )

    assert set(frame["batter"]) == {1, 2}


def test_preload_statcast_cache_refreshes_only_trailing_window(tmp_path, monkeypatch):
    cache = tmp_path / "mlb_statcast_2026.csv"
    calls: list[tuple[pd.Timestamp, pd.Timestamp, bool]] = []

    def fake_fetch(start, end, *, cache, refresh, timeout, deadline_seconds):
        calls.append((start, end, refresh))
        return pd.DataFrame({"game_date": [end.strftime("%Y-%m-%d")]})

    monkeypatch.setattr("src.models.mlb_hr_statcast_features.fetch_statcast", fake_fetch)

    frame = preload_statcast_cache(
        date(2026, 7, 16),
        lookback_days=120,
        refresh_days=10,
        cache=cache,
        deadline_seconds=360,
    )

    assert calls == [
        (pd.Timestamp("2026-03-18"), pd.Timestamp("2026-07-05"), False),
        (pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-15"), True),
    ]
    assert frame.loc[0, "game_date"] == "2026-07-15"
