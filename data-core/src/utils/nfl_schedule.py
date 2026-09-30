"""One NFL forecast cycle, from Tuesday through the following Monday."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

DENVER = ZoneInfo("America/Denver")


def nfl_week_window(anchor: date) -> tuple[date, date]:
    start = anchor - timedelta(days=(anchor.weekday() - 1) % 7)
    return start, start + timedelta(days=6)


def nfl_prediction_cutoff(anchor: date) -> datetime:
    start, _ = nfl_week_window(anchor)
    return datetime.combine(start, datetime.min.time(), tzinfo=DENVER)
