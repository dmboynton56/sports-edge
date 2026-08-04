"""Configurable redraft fantasy football scoring and lineup helpers.

The predictor stores football outcomes (yards, touchdowns, turnovers, and
defensive events), then this module translates those outcomes into any common
league scoring profile. Keeping this calculation independent of the model
means visitors can change scoring without requiring a new model run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ScoringSettings:
    name: str = "Full PPR"
    passing_yards: float = 0.04
    passing_td: float = 4.0
    interception: float = -2.0
    rushing_yards: float = 0.1
    rushing_td: float = 6.0
    reception: float = 1.0
    receiving_yards: float = 0.1
    receiving_td: float = 6.0
    fumble_lost: float = -2.0
    two_point_conversion: float = 2.0
    kick_extra_point: float = 1.0
    field_goal_0_39: float = 3.0
    field_goal_40_49: float = 4.0
    field_goal_50_plus: float = 5.0
    missed_field_goal: float = 0.0
    dst_sack: float = 1.0
    dst_interception: float = 2.0
    dst_fumble_recovery: float = 2.0
    dst_td: float = 6.0
    dst_safety: float = 2.0
    dst_blocked_kick: float = 2.0
    dst_points_0: float = 10.0
    dst_points_1_6: float = 7.0
    dst_points_7_13: float = 4.0
    dst_points_14_20: float = 1.0
    dst_points_21_27: float = 0.0
    dst_points_28_34: float = -1.0
    dst_points_35_plus: float = -4.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "ScoringSettings":
        if not values:
            return cls()
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        normalized = {key: value for key, value in values.items() if key in allowed}
        return cls(**normalized)


FULL_PPR_SCORING = ScoringSettings()
HALF_PPR_SCORING = ScoringSettings(name="Half PPR", reception=0.5)
STANDARD_SCORING = ScoringSettings(name="Standard", reception=0.0)


@dataclass(frozen=True)
class RosterSettings:
    teams: int = 12
    quarterback: int = 1
    running_back: int = 2
    wide_receiver: int = 2
    tight_end: int = 1
    flex: int = 1
    kicker: int = 1
    defense: int = 1
    bench: int = 6

    @property
    def total_roster_slots(self) -> int:
        return (
            self.quarterback
            + self.running_back
            + self.wide_receiver
            + self.tight_end
            + self.flex
            + self.kicker
            + self.defense
            + self.bench
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "RosterSettings":
        if not values:
            return cls()
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        normalized = {
            key: max(0, int(value))
            for key, value in values.items()
            if key in allowed and value is not None
        }
        if "teams" in normalized:
            normalized["teams"] = min(20, max(4, normalized["teams"]))
        return cls(**normalized)


DEFAULT_ROSTER = RosterSettings()


def _number(statline: Mapping[str, Any], key: str) -> float:
    value = statline.get(key, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _field_goal_points(statline: Mapping[str, Any], settings: ScoringSettings) -> float:
    return (
        _number(statline, "fg_made_0_39") * settings.field_goal_0_39
        + _number(statline, "fg_made_40_49") * settings.field_goal_40_49
        + _number(statline, "fg_made_50_plus") * settings.field_goal_50_plus
        + _number(statline, "fg_missed") * settings.missed_field_goal
    )


def _points_allowed_points(statline: Mapping[str, Any], settings: ScoringSettings) -> float:
    points_allowed = _number(statline, "dst_points_allowed")
    if points_allowed <= 0:
        return settings.dst_points_0
    if points_allowed <= 6:
        return settings.dst_points_1_6
    if points_allowed <= 13:
        return settings.dst_points_7_13
    if points_allowed <= 20:
        return settings.dst_points_14_20
    if points_allowed <= 27:
        return settings.dst_points_21_27
    if points_allowed <= 34:
        return settings.dst_points_28_34
    return settings.dst_points_35_plus


def score_statline(
    statline: Mapping[str, Any],
    settings: ScoringSettings = FULL_PPR_SCORING,
    position: str | None = None,
) -> float:
    """Score a player/team statline using a common redraft profile.

    The function accepts both nflverse-style names and the normalized names
    emitted by the fantasy pipeline. Missing fields are treated as zero so a
    partial projection can still be displayed with a quality flag.
    """

    position_key = (position or statline.get("position") or "").upper()
    if position_key == "DST":
        return round(
            _number(statline, "dst_sacks") * settings.dst_sack
            + _number(statline, "dst_interceptions") * settings.dst_interception
            + _number(statline, "dst_fumble_recoveries") * settings.dst_fumble_recovery
            + _number(statline, "dst_tds") * settings.dst_td
            + _number(statline, "dst_safeties") * settings.dst_safety
            + _number(statline, "dst_blocked_kicks") * settings.dst_blocked_kick
            + _points_allowed_points(statline, settings),
            2,
        )
    if position_key == "K":
        return round(
            _number(statline, "extra_points_made") * settings.kick_extra_point
            + _field_goal_points(statline, settings),
            2,
        )

    points = (
        _number(statline, "passing_yards") * settings.passing_yards
        + _number(statline, "passing_tds") * settings.passing_td
        + _number(statline, "interceptions") * settings.interception
        + _number(statline, "rushing_yards") * settings.rushing_yards
        + _number(statline, "rushing_tds") * settings.rushing_td
        + _number(statline, "receptions") * settings.reception
        + _number(statline, "receiving_yards") * settings.receiving_yards
        + _number(statline, "receiving_tds") * settings.receiving_td
        + _number(statline, "fumbles_lost") * settings.fumble_lost
        + _number(statline, "two_point_conversions") * settings.two_point_conversion
    )
    return round(points, 2)


def projection_points(
    projection: Mapping[str, Any], settings: ScoringSettings = FULL_PPR_SCORING
) -> dict[str, float]:
    """Score median/low/high component projections for display."""

    output: dict[str, float] = {}
    for label, prefix in (("floor", "low"), ("median", "median"), ("ceiling", "high")):
        line = {
            key: value
            for key, value in projection.items()
            if not key.endswith("_low") and not key.endswith("_high") and not key.endswith("_median")
        }
        for key, value in projection.items():
            if key.endswith(f"_{prefix}"):
                line[key[: -len(prefix) - 1]] = value
        output[label] = score_statline(line, settings, projection.get("position"))
    return output


def eligible_positions(position: str, slot: str) -> bool:
    position = position.upper()
    slot = slot.upper()
    if slot == "FLEX":
        return position in {"RB", "WR", "TE"}
    return (slot == "QB" and position == "QB") or (
        slot == "RB" and position == "RB"
    ) or (slot == "WR" and position == "WR") or (
        slot == "TE" and position == "TE"
    ) or (slot == "K" and position == "K") or (
        slot in {"DST", "DEF"} and position == "DST"
    )


def starter_slots(roster: RosterSettings) -> list[str]:
    return (
        ["QB"] * roster.quarterback
        + ["RB"] * roster.running_back
        + ["WR"] * roster.wide_receiver
        + ["TE"] * roster.tight_end
        + ["FLEX"] * roster.flex
        + ["K"] * roster.kicker
        + ["DST"] * roster.defense
    )


def replacement_rank(position: str, roster: RosterSettings) -> int:
    """Return the first replacement rank for value-based drafting."""

    position = position.upper()
    demand = {
        "QB": roster.quarterback,
        "RB": roster.running_back,
        "WR": roster.wide_receiver,
        "TE": roster.tight_end,
        "K": roster.kicker,
        "DST": roster.defense,
    }.get(position, 1)
    if position in {"RB", "WR", "TE"}:
        demand += roster.flex
    return max(1, roster.teams * demand + 1)
