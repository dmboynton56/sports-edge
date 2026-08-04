from src.fantasy.scoring import (
    DEFAULT_ROSTER,
    FULL_PPR_SCORING,
    HALF_PPR_SCORING,
    STANDARD_SCORING,
    RosterSettings,
    ScoringSettings,
    projection_points,
    replacement_rank,
    score_statline,
)


def test_full_half_and_standard_reception_scoring():
    line = {"receptions": 8, "receiving_yards": 100, "receiving_tds": 1}
    assert score_statline(line, FULL_PPR_SCORING, "WR") == 24
    assert score_statline(line, HALF_PPR_SCORING, "WR") == 20
    assert score_statline(line, STANDARD_SCORING, "WR") == 16


def test_kicker_and_dst_buckets():
    kicker = {"extra_points_made": 3, "fg_made_0_39": 1, "fg_made_40_49": 1, "fg_made_50_plus": 1}
    assert score_statline(kicker, FULL_PPR_SCORING, "K") == 15
    defense = {
        "dst_sacks": 4,
        "dst_interceptions": 1,
        "dst_fumble_recoveries": 1,
        "dst_tds": 1,
        "dst_safeties": 1,
        "dst_points_allowed": 6,
    }
    assert score_statline(defense, FULL_PPR_SCORING, "DST") == 23


def test_projection_range_scores_and_custom_profile():
    projection = {
        "position": "RB",
        "rushing_yards": 100,
        "rushing_yards_low": 60,
        "rushing_yards_high": 140,
        "rushing_tds": 1,
        "rushing_tds_low": 0,
        "rushing_tds_high": 2,
        "receptions": 5,
        "receptions_low": 2,
        "receptions_high": 8,
    }
    values = projection_points(projection, FULL_PPR_SCORING)
    assert values["floor"] < values["median"] < values["ceiling"]
    custom = ScoringSettings.from_dict({"name": "Custom", "reception": 2})
    assert score_statline({"receptions": 5}, custom, "WR") == 10


def test_roster_and_replacement_rank():
    roster = RosterSettings.from_dict({"teams": 10, "running_back": 2, "flex": 2})
    assert roster.teams == 10
    assert replacement_rank("RB", roster) == 41
    assert DEFAULT_ROSTER.total_roster_slots == 15
