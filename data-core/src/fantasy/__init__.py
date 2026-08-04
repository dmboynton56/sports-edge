"""Fantasy football scoring, projections, and roster-planning utilities."""

from .scoring import (
    DEFAULT_ROSTER,
    FULL_PPR_SCORING,
    HALF_PPR_SCORING,
    STANDARD_SCORING,
    RosterSettings,
    ScoringSettings,
    score_statline,
)

__all__ = [
    "DEFAULT_ROSTER",
    "FULL_PPR_SCORING",
    "HALF_PPR_SCORING",
    "STANDARD_SCORING",
    "RosterSettings",
    "ScoringSettings",
    "score_statline",
]
