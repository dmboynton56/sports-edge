"""Pure MLB HR trusted-board contract helpers.

The workflow and the tests share these rules so a dashboard count cannot
quietly disagree with the publishing job.  The functions intentionally accept
plain mappings; the sync step can pass provider JSON or pandas records without
coupling the contract to a database driver.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

DENVER = ZoneInfo("America/Denver")
MAX_ODDS_AGE = timedelta(minutes=60)
MIN_TOP25_COVERAGE = 0.80


def parse_timestamp(value: Any) -> datetime | None:
    """Parse ISO timestamps and normalize them to aware UTC datetimes."""

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_slate_date(now: datetime | None = None) -> date:
    """Return the MLB slate date in America/Denver, including DST changes."""

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(DENVER).date()


def run_window_for(now: datetime | None = None, manual: bool = False) -> str:
    """Classify a run as morning, afternoon, or explicitly manual."""

    if manual:
        return "manual"
    current = (now or datetime.now(timezone.utc)).astimezone(DENVER)
    return "afternoon" if current.hour >= 12 else "morning"


def _numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def is_priced_row(
    row: Mapping[str, Any],
    publication_ts: datetime | str,
    *,
    max_age: timedelta = MAX_ODDS_AGE,
) -> bool:
    """Whether an odds snapshot satisfies the immutable priced-row contract."""

    price = _numeric(row.get("american_price", row.get("best_price", row.get("price"))))
    market_probability = _numeric(
        row.get("market_probability", row.get("no_vig_probability", row.get("implied_probability")))
    )
    book = row.get("book") or row.get("best_book")
    odds_ts = parse_timestamp(row.get("odds_snapshot_ts", row.get("snapshot_ts")))
    published = parse_timestamp(publication_ts)
    if price is None or price == 0 or not book or market_probability is None or not 0 < market_probability < 1:
        return False
    if odds_ts is None or published is None or odds_ts > published:
        return False
    return published - odds_ts <= max_age


def filter_unstarted_rows(
    rows: Iterable[Mapping[str, Any]],
    now: datetime | None = None,
    *,
    hide_within: timedelta = timedelta(minutes=5),
) -> list[Mapping[str, Any]]:
    """Keep candidates whose event has not started and is not within five minutes."""

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current.astimezone(timezone.utc) + hide_within
    output: list[Mapping[str, Any]] = []
    for row in rows:
        event_time = parse_timestamp(row.get("event_time", row.get("eventTime")))
        if event_time is not None and event_time > cutoff:
            output.append(row)
    return output


def coverage_stats(
    rows: Iterable[Mapping[str, Any]],
    publication_ts: datetime | str,
    *,
    now: datetime | None = None,
    top_n: int = 25,
) -> dict[str, int | float | None]:
    """Compute priced counts and top-N coverage without a zero/zero trap."""

    candidates = list(rows)
    eligible = filter_unstarted_rows(candidates, now=now)
    eligible.sort(key=lambda row: (_numeric(row.get("rank")) or float("inf"), str(row.get("player_id", ""))))
    denominator = min(top_n, len(eligible))
    priced = [row for row in candidates if is_priced_row(row, publication_ts)]
    top = eligible[:denominator]
    top_priced = sum(1 for row in top if is_priced_row(row, publication_ts))
    coverage = (top_priced / denominator) if denominator else None
    return {
        "total_candidates": len(candidates),
        "priced_candidates": len(priced),
        "top25_denominator": denominator,
        "top25_priced_count": top_priced,
        "top25_coverage": coverage,
    }


def classify_run(
    *,
    has_slate: bool,
    source_ok: bool,
    predictions_valid: bool,
    top25_coverage: float | None,
) -> str:
    """Classify a run according to the public fail-closed status contract."""

    if not has_slate and source_ok and predictions_valid:
        return "no_slate"
    if not source_ok or not predictions_valid:
        return "failed"
    if top25_coverage is not None and top25_coverage >= MIN_TOP25_COVERAGE:
        return "healthy"
    return "partial"


def schedule_confirms_slate_over(
    schedule_rows: Iterable[Mapping[str, Any]],
    expected_game_ids: Iterable[str],
) -> bool:
    """Return true only when the official schedule confirms every slate game is final.

    ``expected_game_ids`` comes from the last non-empty published board.  Requiring
    those games to still appear prevents an empty or partial schedule response from
    being mistaken for a completed slate.
    """

    rows = list(schedule_rows)
    expected = {str(game_id).removeprefix("MLB_") for game_id in expected_game_ids if game_id}
    scheduled = {str(row.get("game_pk")) for row in rows if row.get("game_pk") is not None}
    return bool(rows) and bool(expected) and expected <= scheduled and all(bool(row.get("completed")) for row in rows)


def select_latest_pregame_publication(
    rows: Iterable[Mapping[str, Any]],
    game_start: datetime | str,
) -> Mapping[str, Any] | None:
    """Select the latest published snapshot before first pitch for a player."""

    start = parse_timestamp(game_start)
    if start is None:
        return None
    eligible = []
    for row in rows:
        published = parse_timestamp(row.get("published_at", row.get("prediction_ts")))
        event_time = parse_timestamp(row.get("event_time", row.get("eventTime")))
        if published is None or published > start:
            continue
        if event_time is None or event_time >= start:
            eligible.append((published, row))
    if not eligible:
        return None
    eligible.sort(key=lambda item: item[0])
    return eligible[-1][1]


def american_to_decimal(price: Any) -> float | None:
    value = _numeric(price)
    if value is None or value == 0:
        return None
    return 1 + (value / 100 if value > 0 else 100 / abs(value))


def american_to_implied_probability(price: Any) -> float | None:
    value = _numeric(price)
    if value is None or value == 0:
        return None
    return 100 / (value + 100) if value > 0 else abs(value) / (abs(value) + 100)


def no_vig_probability(over_implied: Any, under_implied: Any) -> float | None:
    over = _numeric(over_implied)
    under = _numeric(under_implied)
    if over is None or under is None or over < 0 or under < 0 or over + under <= 0:
        return None
    return over / (over + under)


def edge(model_probability: Any, market_probability: Any) -> float | None:
    model = _numeric(model_probability)
    market = _numeric(market_probability)
    if model is None or market is None:
        return None
    return model - market


def expected_value(model_probability: Any, decimal_price: Any) -> float | None:
    model = _numeric(model_probability)
    decimal = _numeric(decimal_price)
    if model is None or decimal is None or decimal <= 1:
        return None
    return model * decimal - 1


def quarter_kelly(model_probability: Any, decimal_price: Any) -> float | None:
    model = _numeric(model_probability)
    decimal = _numeric(decimal_price)
    if model is None or decimal is None or decimal <= 1:
        return None
    return max(((model * (decimal - 1) - (1 - model)) / (decimal - 1)) * 0.25, 0.0)


def flat_unit_result(row: Mapping[str, Any]) -> float | None:
    """Return flat one-unit P/L, or None for official void/missing outcomes."""

    status = str(row.get("outcome_status") or row.get("status") or "").lower()
    if status in {"void", "scratch", "scratched", "cancelled", "canceled", "postponed"}:
        return None
    plate_appearances = _numeric(row.get("actual_plate_appearances"))
    actual = row.get("actual_home_run")
    if plate_appearances is None or plate_appearances <= 0 or actual is None:
        return None
    price = american_to_decimal(row.get("american_price", row.get("best_price", row.get("price"))))
    if price is None:
        return None
    return price - 1 if bool(actual) else -1


def summarize_flat_results(rows: Iterable[Mapping[str, Any]]) -> dict[str, int | float | None]:
    """Summarize priced official outcomes while keeping model-only accuracy separate."""

    priced = [row for row in rows if row.get("american_price", row.get("best_price")) not in (None, 0)]
    units = [result for row in priced if (result := flat_unit_result(row)) is not None]
    hits = sum(1 for result in units if result > 0)
    losses = sum(1 for result in units if result < 0 and result == -1)
    model_evaluable = [row for row in rows if _numeric(row.get("actual_plate_appearances")) and _numeric(row.get("actual_plate_appearances")) > 0 and row.get("actual_home_run") is not None]
    model_hits = sum(1 for row in model_evaluable if bool(row.get("actual_home_run")))
    return {
        "priced_sample": len(units),
        "priced_hits": hits,
        "priced_losses": losses,
        "priced_hit_rate": hits / len(units) if units else None,
        "flat_units": sum(units) if units else 0.0,
        "flat_roi": sum(units) / len(units) if units else None,
        "model_only_sample": len(model_evaluable),
        "model_only_hits": model_hits,
        "model_only_hit_rate": model_hits / len(model_evaluable) if model_evaluable else None,
    }


__all__ = [
    "DENVER",
    "MAX_ODDS_AGE",
    "MIN_TOP25_COVERAGE",
    "american_to_decimal",
    "american_to_implied_probability",
    "classify_run",
    "coverage_stats",
    "edge",
    "expected_value",
    "filter_unstarted_rows",
    "flat_unit_result",
    "is_priced_row",
    "parse_timestamp",
    "no_vig_probability",
    "quarter_kelly",
    "resolve_slate_date",
    "run_window_for",
    "schedule_confirms_slate_over",
    "select_latest_pregame_publication",
    "summarize_flat_results",
]
