import pytest

from src.data.mlb_boxscore_fetcher import _parse_game_info, _parse_weather, _team_stats_summary


@pytest.mark.parametrize(
    ("weather", "wind", "expected"),
    [
        (
            "66 degrees, Partly Cloudy.",
            "9 mph, L To R.",
            {
                "temp_f": 66,
                "weather_condition": "Partly Cloudy.",
                "wind_mph": 9,
                "wind_dir": "L To R",
            },
        ),
        (
            "72 degrees, Roof Closed.",
            "0 mph, None.",
            {
                "temp_f": 72,
                "weather_condition": "Roof Closed.",
                "wind_mph": 0,
                "wind_dir": "None",
            },
        ),
    ],
)
def test_parse_weather(weather, wind, expected):
    info = [{"label": "Weather", "value": weather}, {"label": "Wind", "value": wind}]

    assert _parse_weather(info) == expected


def test_parse_weather_missing_labels_returns_none_fields():
    assert _parse_weather([]) == {
        "temp_f": None,
        "weather_condition": None,
        "wind_mph": None,
        "wind_dir": None,
    }


@pytest.mark.parametrize(
    "info",
    [
        [{"label": "Weather", "value": "unknown"}, {"label": "Wind", "value": "variable"}],
        [{"label": "Weather", "value": None}, {"label": "Wind", "value": None}],
        None,
        [None, "not a mapping"],
    ],
)
def test_parse_weather_malformed_values_do_not_raise(info):
    assert _parse_weather(info) == {
        "temp_f": None,
        "weather_condition": None,
        "wind_mph": None,
        "wind_dir": None,
    }


def test_parse_game_info_handles_attendance_and_text_fields():
    info = [
        {"label": "First pitch", "value": "7:12 PM"},
        {"label": "Attendance", "value": "41,229"},
        {"label": "Game Duration", "value": "2:41"},
    ]

    parsed = _parse_game_info(info)

    assert parsed["first_pitch"] == "7:12 PM"
    assert parsed["attendance"] == 41229
    assert parsed["game_duration"] == "2:41"


def test_parse_game_info_malformed_attendance_is_none():
    assert _parse_game_info([{"label": "Attendance", "value": "not available"}])["attendance"] is None


def test_parse_game_info_supports_boxscore_abbreviations():
    parsed = _parse_game_info(
        [
            {"label": "Att", "value": "45,568."},
            {"label": "T", "value": "3:05."},
        ]
    )

    assert parsed["attendance"] == 45568
    assert parsed["game_duration"] == "3:05."


def test_team_stats_summary_coerces_missing_stats_to_zero():
    team_payload = {
        "teamStats": {
            "batting": {
                "strikeOuts": ".---",
                "baseOnBalls": None,
                "hits": "8",
                "homeRuns": 2.0,
            },
            "pitching": {"strikeOuts": "11"},
        }
    }

    assert _team_stats_summary(team_payload, "home") == {
        "home_team_strikeouts": 0,
        "home_team_walks": 0,
        "home_team_hits": 8,
        "home_team_home_runs": 2,
        "home_team_pitching_strikeouts": 11,
    }
