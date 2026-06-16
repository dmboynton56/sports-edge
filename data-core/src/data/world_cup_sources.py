"""World Cup source normalization and model-input builders.

The prediction model consumes two simple CSVs: team ratings and fixtures. This
module keeps the messy source-specific parsing outside the model and makes each
feature contribution inspectable before it is loaded into BigQuery or Supabase.
"""

from __future__ import annotations

from datetime import date, datetime
from io import StringIO
import json
from pathlib import Path
import re
from typing import Iterable, Optional, Sequence
import unicodedata

import numpy as np
import pandas as pd
import requests


ESPN_WORLD_CUP_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
WORLD_FOOTBALL_ELO_TSV_URL = "https://www.eloratings.net/World.tsv"
WORLD_FOOTBALL_ELO_TEAMS_URL = "https://www.eloratings.net/en.teams.tsv"
HOST_TEAMS_2026 = ("Canada", "Mexico", "United States")
PLACEHOLDER_TEAMS = {"", "TBD", "To Be Determined", "Winner Match", "Runner-up"}
CANONICAL_TEAM_ALIASES = {
    "bosnia herzegovina": "bosnia and herzegovina",
    "congo dr": "dr congo",
    "cote divoire": "ivory coast",
    "curacao": "curacao",
    "korea republic": "south korea",
    "turkiye": "turkey",
    "usa": "united states",
    "united states of america": "united states",
}

MODEL_TEAM_COLUMNS = [
    "team",
    "group",
    "fifa_rank",
    "elo",
    "form_points_per_game",
    "form_goal_diff_per_game",
    "world_cup_experience_score",
    "star_player_score",
    "host_boost",
    "market_rating",
]

FIXTURE_COLUMNS = [
    "match_id",
    "stage",
    "group",
    "kickoff_utc",
    "home_team",
    "away_team",
    "status",
    "home_score",
    "away_score",
    "neutral_site",
    "venue",
    "source",
    "raw_record",
]


def _canonical(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip())
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    text = " ".join(text.split())
    return CANONICAL_TEAM_ALIASES.get(text, text)


def _is_placeholder_team(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if text in PLACEHOLDER_TEAMS:
        return True
    lowered = text.lower()
    return (
        lowered == "tbd"
        or " winner" in lowered
        or "2nd place" in lowered
        or lowered.startswith("winner ")
        or lowered.startswith("group ")
        or lowered.startswith("third place group ")
        or lowered.startswith("round of ")
        or lowered.startswith("runner-up ")
        or lowered.startswith("runner up ")
        or lowered.startswith("loser ")
    )


def _format_espn_date(value: object) -> str:
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y%m%d")
    return pd.to_datetime(value).strftime("%Y%m%d")


def _first_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    normalized = {_canonical(col).replace(" ", "_"): col for col in df.columns}
    for name in names:
        key = _canonical(name).replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def _optional_int(value: object) -> object:
    if value is None or pd.isna(value) or value == "":
        return pd.NA
    return int(float(value))


def _optional_float(value: object) -> object:
    if value is None or pd.isna(value) or value == "":
        return np.nan
    return float(value)


def _bool_value(value: object, default: bool = True) -> bool:
    if value is None or pd.isna(value) or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "f", "no", "n"}


def _status_from_espn(competition: dict) -> str:
    status_type = (competition.get("status") or {}).get("type") or {}
    if status_type.get("completed"):
        return "final"
    state = str(status_type.get("state") or "").lower()
    if state in {"in", "pre", "post"}:
        return {"in": "in_progress", "pre": "scheduled", "post": "final"}[state]
    return str(status_type.get("description") or "scheduled").lower().replace(" ", "_")


def _stage_from_espn(event: dict, competition: dict) -> str:
    candidates = [
        (competition.get("stage") or {}).get("description"),
        (event.get("season") or {}).get("slug"),
        (event.get("season") or {}).get("type"),
        competition.get("altGameNote"),
        event.get("shortName"),
    ]
    text = " ".join(str(value).lower() for value in candidates if value)
    if "final" in text and "semi" not in text:
        return "final"
    if "semi" in text:
        return "semifinal"
    if "quarter" in text:
        return "quarterfinal"
    if "round of 16" in text:
        return "round_of_16"
    if "round of 32" in text:
        return "round_of_32"
    return "group"


def _stage_from_placeholder_matchup(home_team: str, away_team: str, fallback: str) -> str:
    matchup = f"{home_team} {away_team}".lower()
    if "round of 16" in matchup:
        return "quarterfinal"
    if "round of 32" in matchup:
        return "round_of_16"
    if "group " in matchup or "third place group" in matchup:
        return "round_of_32"
    return fallback


def _group_from_espn(competition: dict) -> Optional[str]:
    note = str(competition.get("altGameNote") or "")
    marker = "Group "
    if marker not in note:
        return None
    group = note.split(marker, 1)[1].split(",", 1)[0].strip()
    return group or None


def _team_name_from_competitor(competitor: dict) -> str:
    team = competitor.get("team") or {}
    return str(
        team.get("displayName")
        or team.get("name")
        or team.get("shortDisplayName")
        or competitor.get("displayName")
        or competitor.get("name")
        or ""
    )


def _score_from_competitor(competitor: dict) -> object:
    score = competitor.get("score")
    if score is None or score == "":
        return pd.NA
    try:
        return int(float(score))
    except (TypeError, ValueError):
        return pd.NA


def parse_espn_world_cup_scoreboard(payload: dict, *, season: int = 2026) -> pd.DataFrame:
    """Normalize an ESPN soccer scoreboard response into fixture rows."""

    rows = []
    for event in payload.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        competitors = competition.get("competitors") or []
        if len(competitors) < 2:
            continue

        home = next((item for item in competitors if item.get("homeAway") == "home"), competitors[0])
        away = next((item for item in competitors if item.get("homeAway") == "away"), competitors[1])
        match_id = str(event.get("id") or competition.get("id") or "")
        if not match_id:
            continue

        venue = competition.get("venue") or {}
        venue_name = venue.get("fullName") or venue.get("displayName") or venue.get("name")
        kickoff = competition.get("date") or event.get("date")
        kickoff_utc = pd.to_datetime(kickoff, utc=True).isoformat().replace("+00:00", "Z") if kickoff else None
        status = _status_from_espn(competition)
        home_score = _score_from_competitor(home) if status == "final" else pd.NA
        away_score = _score_from_competitor(away) if status == "final" else pd.NA
        group_name = _group_from_espn(competition)
        home_team = _team_name_from_competitor(home)
        away_team = _team_name_from_competitor(away)
        stage = _stage_from_placeholder_matchup(
            home_team,
            away_team,
            _stage_from_espn(event, competition),
        )

        rows.append(
            {
                "match_id": match_id,
                "external_match_id": match_id,
                "season": int(season),
                "tournament": "FIFA World Cup",
                "stage": stage,
                "group": group_name,
                "group_name": group_name,
                "kickoff_utc": kickoff_utc,
                "home_team": home_team,
                "away_team": away_team,
                "status": status,
                "home_score": home_score,
                "away_score": away_score,
                "neutral_site": True,
                "venue": venue_name,
                "source": "espn_scoreboard",
                "raw_record": json.dumps(event, sort_keys=True),
            }
        )

    return pd.DataFrame(rows)


def fetch_espn_world_cup_fixtures(
    *,
    start_date: Optional[object] = None,
    end_date: Optional[object] = None,
    season: int = 2026,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch World Cup fixtures/results from ESPN's no-key soccer scoreboard."""

    params = {}
    if start_date and end_date:
        params["dates"] = f"{_format_espn_date(start_date)}-{_format_espn_date(end_date)}"
    response = requests.get(
        ESPN_WORLD_CUP_SCOREBOARD_URL,
        params=params,
        headers={"User-Agent": "SportsEdge/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_espn_world_cup_scoreboard(response.json(), season=season)


def normalize_world_football_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Return `team`, `elo`, and optional `elo_rank` from World Football Elo data."""

    data = df.copy()
    if not {"team", "elo"}.issubset({_canonical(col).replace(" ", "_") for col in data.columns}):
        if len(data.columns) >= 3:
            data = data.rename(columns={data.columns[0]: "elo_rank", data.columns[1]: "team", data.columns[2]: "elo"})

    team_col = _first_column(data, ["team", "country", "nation", "name"])
    elo_col = _first_column(data, ["elo", "rating", "points"])
    rank_col = _first_column(data, ["elo_rank", "rank", "rk"])
    if not team_col or not elo_col:
        raise ValueError("World Football Elo data must include team/country and elo/rating columns")

    out = pd.DataFrame(
        {
            "team": data[team_col].astype(str),
            "elo": pd.to_numeric(data[elo_col], errors="coerce"),
        }
    )
    if rank_col:
        out["elo_rank"] = pd.to_numeric(data[rank_col], errors="coerce").astype("Int64")
    return out.dropna(subset=["team", "elo"]).drop_duplicates(subset=["team"], keep="first")


def parse_world_football_elo_tsv(ratings_tsv: str, teams_tsv: Optional[str] = None) -> pd.DataFrame:
    """Parse World Football Elo's current `World.tsv` format.

    The public site serves ratings as a headerless TSV where the team field is a
    two-letter code. `en.teams.tsv` maps those codes to display names.
    """

    raw = pd.read_csv(StringIO(ratings_tsv), sep="\t", header=None)
    if raw.empty or raw.shape[1] < 4:
        raise ValueError("World Football Elo TSV did not contain the expected rating columns")

    team_map: dict[str, str] = {}
    if teams_tsv:
        for line in teams_tsv.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                team_map[parts[0]] = parts[1]

    out = pd.DataFrame(
        {
            "elo_rank": pd.to_numeric(raw.iloc[:, 0], errors="coerce").astype("Int64"),
            "team_code": raw.iloc[:, 2].astype(str),
            "elo": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
        }
    )
    out["team"] = out["team_code"].map(team_map).fillna(out["team_code"])
    return out[["team", "elo", "elo_rank"]].dropna(subset=["team", "elo"]).drop_duplicates(subset=["team"])


def fetch_world_football_elo(
    url: str = WORLD_FOOTBALL_ELO_TSV_URL,
    *,
    teams_url: str = WORLD_FOOTBALL_ELO_TEAMS_URL,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch current World Football Elo ratings from the public TSV files."""

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "SportsEdge/1.0"})
    response.raise_for_status()
    if url.endswith("World.tsv") or not response.text.lower().startswith(("team", "country", "rank")):
        teams_response = requests.get(teams_url, timeout=timeout, headers={"User-Agent": "SportsEdge/1.0"})
        teams_response.raise_for_status()
        return parse_world_football_elo_tsv(response.text, teams_response.text)

    raw = pd.read_csv(StringIO(response.text), sep="\t")
    return normalize_world_football_elo(raw)


def extract_espn_team_form(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Extract team recent-form strings from ESPN raw fixture records."""

    if fixtures.empty or "raw_record" not in fixtures.columns:
        return pd.DataFrame(columns=["team", "form_points_per_game"])

    rows = []
    for raw_record in fixtures["raw_record"].dropna():
        try:
            event = json.loads(raw_record) if isinstance(raw_record, str) else raw_record
        except (TypeError, json.JSONDecodeError):
            continue
        for competition in event.get("competitions") or []:
            for competitor in competition.get("competitors") or []:
                team = _team_name_from_competitor(competitor)
                form = str(competitor.get("form") or "").upper()
                if _is_placeholder_team(team) or not form:
                    continue
                values = [3 if char == "W" else 1 if char == "D" else 0 for char in form if char in {"W", "D", "L"}]
                if values:
                    rows.append({"team": team, "form_points_per_game": float(np.mean(values))})

    if not rows:
        return pd.DataFrame(columns=["team", "form_points_per_game"])
    return pd.DataFrame(rows).groupby("team", as_index=False)["form_points_per_game"].mean()


def normalize_fifa_rankings(df: pd.DataFrame) -> pd.DataFrame:
    team_col = _first_column(df, ["team", "country", "nation", "name"])
    rank_col = _first_column(df, ["fifa_rank", "rank", "rk"])
    points_col = _first_column(df, ["fifa_points", "points", "total_points"])
    if not team_col:
        raise ValueError("FIFA ranking data must include a team/country column")

    out = pd.DataFrame({"team": df[team_col].astype(str)})
    if rank_col:
        out["fifa_rank"] = pd.to_numeric(df[rank_col], errors="coerce")
    elif points_col:
        out["fifa_rank"] = pd.to_numeric(df[points_col], errors="coerce").rank(method="first", ascending=False)
    else:
        raise ValueError("FIFA ranking data must include rank or points")
    if points_col:
        out["fifa_points"] = pd.to_numeric(df[points_col], errors="coerce")
    return out.dropna(subset=["team", "fifa_rank"]).drop_duplicates(subset=["team"], keep="first")


def build_recent_team_form(
    results: pd.DataFrame,
    *,
    teams: Optional[Iterable[str]] = None,
    as_of_date: Optional[object] = None,
    lookback_matches: int = 10,
) -> pd.DataFrame:
    """Build recent points-per-game and goal-difference form from match results."""

    if results.empty:
        return pd.DataFrame(columns=["team", "form_points_per_game", "form_goal_diff_per_game"])

    required = {"home_team", "away_team", "home_score", "away_score"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Recent results missing columns: {sorted(missing)}")

    data = results.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        if as_of_date is not None:
            data = data[data["date"] <= pd.to_datetime(as_of_date)]
        data = data.sort_values("date", ascending=False)

    if teams is None:
        team_names = sorted(set(data["home_team"].astype(str)) | set(data["away_team"].astype(str)))
    else:
        team_names = sorted({str(team) for team in teams})

    rows = []
    for team in team_names:
        mask = (data["home_team"].astype(str) == team) | (data["away_team"].astype(str) == team)
        recent = data[mask].head(lookback_matches)
        points = []
        goal_diffs = []
        for row in recent.to_dict(orient="records"):
            try:
                home_score = int(row["home_score"])
                away_score = int(row["away_score"])
            except (TypeError, ValueError):
                continue
            is_home = str(row["home_team"]) == team
            goals_for = home_score if is_home else away_score
            goals_against = away_score if is_home else home_score
            points.append(3 if goals_for > goals_against else 1 if goals_for == goals_against else 0)
            goal_diffs.append(goals_for - goals_against)

        if points:
            rows.append(
                {
                    "team": team,
                    "form_points_per_game": float(np.mean(points)),
                    "form_goal_diff_per_game": float(np.mean(goal_diffs)),
                }
            )

    return pd.DataFrame(rows)


_STAGE_EXPERIENCE = {
    "champion": 12.0,
    "winner": 12.0,
    "final": 10.0,
    "runner_up": 10.0,
    "semifinal": 8.0,
    "quarterfinal": 6.0,
    "round_of_16": 4.0,
    "round_of_32": 2.5,
    "group": 1.0,
}


def _stage_experience_value(stage: object) -> float:
    key = _canonical(stage).replace(" ", "_").replace("-", "_")
    return _STAGE_EXPERIENCE.get(key, 1.0 if key else 0.0)


def build_world_cup_experience(history: pd.DataFrame, *, season: int = 2026) -> pd.DataFrame:
    """Build a recency-decayed World Cup experience score from team history."""

    if history.empty:
        return pd.DataFrame(columns=["team", "world_cup_experience_score"])
    if "world_cup_experience_score" in history.columns:
        team_col = _first_column(history, ["team", "country", "nation"])
        if not team_col:
            raise ValueError("World Cup history score data must include a team column")
        return history[[team_col, "world_cup_experience_score"]].rename(columns={team_col: "team"})

    team_col = _first_column(history, ["team", "country", "nation"])
    stage_col = _first_column(history, ["stage", "finish", "finish_stage", "best_finish"])
    year_col = _first_column(history, ["season", "year", "tournament_year"])
    if not team_col or not stage_col:
        raise ValueError("World Cup history must include team and stage/finish columns")

    rows = []
    for row in history.to_dict(orient="records"):
        year = int(row[year_col]) if year_col and not pd.isna(row.get(year_col)) else season
        cycles_back = max((season - year) / 4.0, 0.0)
        decay = 0.74 ** cycles_back
        rows.append(
            {
                "team": str(row[team_col]),
                "experience_piece": _stage_experience_value(row.get(stage_col)) * decay,
            }
        )
    scored = pd.DataFrame(rows)
    return (
        scored.groupby("team", as_index=False)["experience_piece"]
        .sum()
        .rename(columns={"experience_piece": "world_cup_experience_score"})
    )


def build_star_player_scores(player_form: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player form/availability into a top-player team score."""

    if player_form.empty:
        return pd.DataFrame(columns=["team", "star_player_score"])
    team_col = _first_column(player_form, ["team", "country", "nation"])
    if not team_col:
        raise ValueError("Player form data must include a team column")

    data = player_form.copy()
    minutes_col = _first_column(data, ["minutes", "club_minutes", "minutes_90s"])
    goals_col = _first_column(data, ["goals", "club_goals"])
    assists_col = _first_column(data, ["assists", "club_assists"])
    xg_col = _first_column(data, ["xg", "club_xg"])
    xa_col = _first_column(data, ["xa", "club_xa"])
    rating_col = _first_column(data, ["rating", "player_rating", "impact_rating"])
    market_col = _first_column(data, ["market_value", "market_value_eur", "transfer_value"])
    availability_col = _first_column(data, ["availability", "available_share", "availability_weight"])
    status_col = _first_column(data, ["status", "injury_status"])

    def values(col: Optional[str]) -> pd.Series:
        if not col:
            return pd.Series(0.0, index=data.index)
        return pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    minutes = values(minutes_col)
    goals = values(goals_col)
    assists = values(assists_col)
    xg = values(xg_col)
    xa = values(xa_col)
    rating = values(rating_col)
    market = values(market_col)

    availability = pd.Series(1.0, index=data.index)
    if availability_col:
        availability = pd.to_numeric(data[availability_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    elif status_col:
        status = data[status_col].astype(str).str.lower()
        availability = pd.Series(
            np.select(
                [status.str.contains("out"), status.str.contains("doubt"), status.str.contains("question")],
                [0.0, 0.25, 0.5],
                default=1.0,
            ),
            index=data.index,
        )

    rating_z = (rating - rating.mean()) / rating.std(ddof=0) if rating.std(ddof=0) > 0 else rating * 0.0
    market_log = np.log1p(market.clip(lower=0.0))
    market_z = (market_log - market_log.mean()) / market_log.std(ddof=0) if market_log.std(ddof=0) > 0 else market_log * 0.0
    data["_impact"] = availability * (
        0.25 * (minutes.clip(lower=0.0) / 900.0)
        + 0.30 * (goals + 0.7 * assists)
        + 0.22 * (xg + xa)
        + 0.15 * rating_z.fillna(0.0)
        + 0.08 * market_z.fillna(0.0)
    )
    top_players = data.sort_values([team_col, "_impact"], ascending=[True, False]).groupby(team_col).head(5)
    scores = top_players.groupby(team_col, as_index=False)["_impact"].sum()
    if not scores.empty and scores["_impact"].std(ddof=0) > 0:
        scores["_impact"] = (scores["_impact"] - scores["_impact"].mean()) / scores["_impact"].std(ddof=0)
    return scores.rename(columns={team_col: "team", "_impact": "star_player_score"})


def build_market_ratings(odds: pd.DataFrame) -> pd.DataFrame:
    """Convert futures odds into a centered market rating prior."""

    if odds.empty:
        return pd.DataFrame(columns=["team", "market_rating"])
    team_col = _first_column(odds, ["team", "country", "nation"])
    prob_col = _first_column(odds, ["implied_prob", "implied_probability", "title_prob"])
    decimal_col = _first_column(odds, ["decimal_odds", "odds_decimal"])
    american_col = _first_column(odds, ["american_odds", "odds_american"])
    if not team_col:
        raise ValueError("Market odds data must include a team column")

    if prob_col:
        implied = pd.to_numeric(odds[prob_col], errors="coerce")
    elif decimal_col:
        implied = 1.0 / pd.to_numeric(odds[decimal_col], errors="coerce")
    elif american_col:
        american = pd.to_numeric(odds[american_col], errors="coerce")
        implied = np.where(american > 0, 100.0 / (american + 100.0), -american / (-american + 100.0))
        implied = pd.Series(implied, index=odds.index)
    else:
        raise ValueError("Market odds data must include implied probability, decimal odds, or American odds")

    out = pd.DataFrame({"team": odds[team_col].astype(str), "implied": implied})
    out = out.dropna(subset=["implied"])
    if out.empty:
        return pd.DataFrame(columns=["team", "market_rating"])
    out["market_rating"] = np.log(out["implied"].clip(lower=1e-6) / out["implied"].mean()).clip(-2.0, 2.0)
    return out[["team", "market_rating"]].drop_duplicates(subset=["team"], keep="first")


def build_team_rating_inputs(
    *,
    teams: Optional[pd.DataFrame] = None,
    fixtures: Optional[pd.DataFrame] = None,
    fifa_rankings: Optional[pd.DataFrame] = None,
    world_elo: Optional[pd.DataFrame] = None,
    recent_results: Optional[pd.DataFrame] = None,
    team_form: Optional[pd.DataFrame] = None,
    world_cup_history: Optional[pd.DataFrame] = None,
    player_form: Optional[pd.DataFrame] = None,
    market_odds: Optional[pd.DataFrame] = None,
    host_teams: Sequence[str] = HOST_TEAMS_2026,
    season: int = 2026,
    as_of_date: Optional[object] = None,
    lookback_matches: int = 10,
) -> pd.DataFrame:
    """Blend source tables into the team-rating CSV consumed by the model."""

    base_rows = []
    if teams is not None and not teams.empty:
        team_col = _first_column(teams, ["team", "country", "nation", "name"])
        if not team_col:
            raise ValueError("Teams data must include a team column")
        group_col = _first_column(teams, ["group", "group_name"])
        for row in teams.to_dict(orient="records"):
            base_rows.append({"team": str(row[team_col]), "group": row.get(group_col) if group_col else None})
    elif fixtures is not None and not fixtures.empty:
        for col in ("home_team", "away_team"):
            for team in fixtures[col].dropna().astype(str).unique():
                if not _is_placeholder_team(team):
                    team_fixtures = fixtures[(fixtures["home_team"] == team) | (fixtures["away_team"] == team)]
                    groups = sorted(
                        {
                            str(group)
                            for group in team_fixtures.get("group", pd.Series(dtype=object)).dropna().tolist()
                            if str(group).strip()
                        }
                    )
                    base_rows.append({"team": team, "group": groups[0] if groups else None})
    else:
        raise ValueError("Provide teams or fixtures to seed World Cup team ratings")

    base = pd.DataFrame(base_rows).drop_duplicates(subset=["team"], keep="first")
    base["_team_key"] = base["team"].map(_canonical)

    def merge_source(source: pd.DataFrame, columns: Sequence[str]) -> None:
        nonlocal base
        if source.empty:
            return
        data = source.copy()
        data["_team_key"] = data["team"].map(_canonical)
        keep = ["_team_key"]
        rename_map = {}
        for col in columns:
            if col not in data.columns:
                continue
            if col in base.columns:
                rename_map[col] = f"{col}__source"
                keep.append(col)
            else:
                keep.append(col)
        base = base.merge(data[keep].rename(columns=rename_map), how="left", on="_team_key")
        for original, source_col in rename_map.items():
            base[original] = base[original].combine_first(base[source_col])
            base = base.drop(columns=[source_col])

    if fifa_rankings is not None:
        merge_source(normalize_fifa_rankings(fifa_rankings), ["fifa_rank"])
    if world_elo is not None:
        merge_source(normalize_world_football_elo(world_elo), ["elo"])
    if recent_results is not None:
        form = build_recent_team_form(
            recent_results,
            teams=base["team"].tolist(),
            as_of_date=as_of_date,
            lookback_matches=lookback_matches,
        )
        merge_source(form, ["form_points_per_game", "form_goal_diff_per_game"])
    if team_form is not None:
        merge_source(team_form, ["form_points_per_game"])
    if world_cup_history is not None:
        merge_source(build_world_cup_experience(world_cup_history, season=season), ["world_cup_experience_score"])
    if player_form is not None:
        merge_source(build_star_player_scores(player_form), ["star_player_score"])
    if market_odds is not None:
        merge_source(build_market_ratings(market_odds), ["market_rating"])

    host_keys = {_canonical(team) for team in host_teams}
    if "host_boost" not in base.columns:
        base["host_boost"] = np.where(base["_team_key"].isin(host_keys), 0.18, 0.0)

    def fill_numeric_column(col: str, default: float) -> None:
        if col not in base.columns:
            base[col] = default
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(default)

    fill_numeric_column("form_points_per_game", 1.5)
    fill_numeric_column("form_goal_diff_per_game", 0.0)
    fill_numeric_column("world_cup_experience_score", 0.0)
    fill_numeric_column("star_player_score", 0.0)
    fill_numeric_column("market_rating", 0.0)

    for col in ["fifa_rank", "elo"]:
        if col not in base.columns:
            base[col] = np.nan
    return base[MODEL_TEAM_COLUMNS].sort_values(["group", "team"], na_position="last").reset_index(drop=True)


def normalize_fixtures_for_model(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return fixture columns expected by `predict_world_cup.py`."""

    if fixtures.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    data = fixtures.copy()
    if "match_id" not in data.columns and "external_match_id" in data.columns:
        data["match_id"] = data["external_match_id"]
    required = {"match_id", "stage", "home_team", "away_team"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"World Cup fixtures missing columns: {sorted(missing)}")
    for col in FIXTURE_COLUMNS:
        if col not in data.columns:
            data[col] = None
    data["status"] = data["status"].fillna("scheduled")
    data["neutral_site"] = data["neutral_site"].map(lambda value: _bool_value(value, default=True))
    data["home_score"] = data["home_score"].map(_optional_int)
    data["away_score"] = data["away_score"].map(_optional_int)
    return data[FIXTURE_COLUMNS].sort_values(["kickoff_utc", "match_id"], na_position="last").reset_index(drop=True)


def _slot_from_placeholder(value: object) -> Optional[str]:
    text = str(value or "").strip()
    lowered = text.lower()
    if lowered.startswith("group ") and lowered.endswith(" winner"):
        group = text.split()[1].upper()
        return f"1{group}"
    if lowered.startswith("group ") and "2nd place" in lowered:
        group = text.split()[1].upper()
        return f"2{group}"
    if lowered.startswith("third place group "):
        return "3*"
    return None


def derive_round_of_32_slots(fixtures: pd.DataFrame) -> list[tuple[str, str]]:
    """Derive simulator slot labels from unresolved ESPN Round of 32 fixtures."""

    if fixtures.empty or not {"stage", "home_team", "away_team"}.issubset(fixtures.columns):
        return []
    rounds = fixtures[fixtures["stage"].astype(str).str.lower() == "round_of_32"].copy()
    if rounds.empty:
        return []
    rounds = rounds.sort_values(["kickoff_utc", "match_id"], na_position="last")
    slots: list[tuple[str, str]] = []
    for row in rounds.to_dict(orient="records"):
        left = _slot_from_placeholder(row.get("home_team"))
        right = _slot_from_placeholder(row.get("away_team"))
        if not left or not right:
            return []
        slots.append((left, right))
    return slots if len(slots) == 16 else []


def write_world_cup_inputs(
    *,
    team_ratings: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams_csv: Path,
    fixtures_csv: Path,
    round_of_32_slots_json: Optional[Path] = None,
) -> None:
    teams_csv.parent.mkdir(parents=True, exist_ok=True)
    fixtures_csv.parent.mkdir(parents=True, exist_ok=True)
    team_ratings.to_csv(teams_csv, index=False)
    fixtures.to_csv(fixtures_csv, index=False)
    if round_of_32_slots_json:
        round_of_32_slots_json.parent.mkdir(parents=True, exist_ok=True)
        slots = derive_round_of_32_slots(fixtures)
        round_of_32_slots_json.write_text(json.dumps(slots, indent=2) + "\n")
