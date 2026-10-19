"""Stats-only NFL receiving fair odds: residual mean + calibrated NB.

Mean is season-to-date / trailing-8 yards (or receptions). Opportunity
(target share, snaps, v2-implied pass volume) and injury/availability are
residual adjustments, not a replacement mean.

Analysis only. Does not write serving tables or live predictions.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.nfl_prop_markets import DFS_BOOK_KEYS, classify_book, compact_closing_rows
from src.models.nfl_receiving_yards import (
    DEFAULT_ALPHA,
    EVAL_SEASONS,
    MIN_MU,
    TARGET_POSITIONS,
    VOLUME_CLIP,
    _lagged_expanding,
    _lagged_rolling,
    apply_opportunity_model,
    attach_distribution,
    crps_nb,
    estimate_dispersion,
    evaluate_predictions,
    implied_team_points,
    p_over_line,
    pinball_loss,
    predictive_quantile,
)
from src.models.nfl_v2 import normalize_team_name
from src.utils.team_codes import NFL_TEAM_NAME_TO_ABBR

RECENCY_WEIGHT_STABLE = 0.28
RECENCY_WEIGHT_EARLY = 0.50
OPP_RESIDUAL_WEIGHT = 0.16
VOLUME_RESIDUAL_WEIGHT = 0.10
RESIDUAL_CLIP = (0.82, 1.18)
SEASON_PRIOR_STRENGTH = 6.0
CHECK_DATA_ABS_P = 0.20
AVAIL_OUT = 0.20
AVAIL_DOUBTFUL = 0.50
AVAIL_QUESTIONABLE = 0.88
RECEPTION_LINES = (2.5, 4.5, 6.5)
YARD_LINES = (24.5, 49.5, 74.5)
PREFERRED_BOOKS = ("draftkings", "fanduel", "pinnacle", "betmgm", "bovada")
_TEAM_SUFFIX = re.compile(r"\s*\(([A-Za-z]{2,3})\)\s*$")


def display_player_name(name: object) -> str:
    """Strip book suffixes like 'Christian McCaffrey (SF)'."""

    return _TEAM_SUFFIX.sub("", str(name or "")).strip()


def _norm_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", display_player_name(name).lower()).strip()


def _team_abbr(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return NFL_TEAM_NAME_TO_ABBR.get(raw.lower()) or normalize_team_name(raw)


def player_on_event(team: object, home: object, away: object) -> bool:
    team_abbr = _team_abbr(team)
    sides = {_team_abbr(home), _team_abbr(away)} - {"", None}
    return bool(team_abbr) and team_abbr in sides


def implied_prob_american(price: object) -> float | None:
    try:
        american = float(price)
    except (TypeError, ValueError):
        return None
    if american == 0 or not math.isfinite(american):
        return None
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def probability_to_american(prob: object) -> int | None:
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or p <= 0.0 or p >= 1.0:
        return None
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def no_vig_over_prob(over_price: object, under_price: object) -> float | None:
    over = implied_prob_american(over_price)
    under = implied_prob_american(under_price)
    if over is None or under is None:
        return None
    total = over + under
    if total <= 0:
        return None
    return float(over / total)


def injury_availability(status: object) -> float:
    text = str(status or "").strip().lower()
    if not text or text in {"active", "probable", ""}:
        return 1.0
    if text in {"out", "ir", "injured reserve", "pup", "suspension", "sus", "inactive"}:
        return AVAIL_OUT
    if "doubt" in text:
        return AVAIL_DOUBTFUL
    if "question" in text:
        return AVAIL_QUESTIONABLE
    return 1.0


def attach_stat_history(features: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Lagged season / trailing-8 / career means for one box-score stat."""

    frame = features.copy()
    if stat not in frame.columns:
        frame[stat] = np.nan
    frame[stat] = pd.to_numeric(frame[stat], errors="coerce")
    sort_cols = [c for c in ("player_id", "season", "week", "game_id") if c in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols)
    frame[f"rolling_{stat}_8"] = _lagged_rolling(frame, stat, 8)
    frame[f"career_{stat}"] = _lagged_expanding(frame, stat)
    if {"player_id", "season"}.issubset(frame.columns):
        frame["season_games_before"] = frame.groupby(["player_id", "season"], sort=False).cumcount().astype(float)
        shifted = frame.groupby(["player_id", "season"], sort=False)[stat].shift(1)
        frame[f"season_avg_{stat}"] = shifted.groupby([frame["player_id"], frame["season"]]).transform(
            lambda values: values.expanding(min_periods=1).mean()
        )
        prior = (
            frame.groupby(["player_id", "season"], sort=False)[stat]
            .mean()
            .rename("_prior_season_mean")
            .reset_index()
        )
        prior["season"] = prior["season"] + 1
        frame = frame.merge(prior, on=["player_id", "season"], how="left")
        frame["prior_season_mean"] = pd.to_numeric(frame["_prior_season_mean"], errors="coerce")
        frame = frame.drop(columns=["_prior_season_mean"])
    else:
        frame["season_games_before"] = 0.0
        frame[f"season_avg_{stat}"] = np.nan
        frame["prior_season_mean"] = np.nan
    return frame


def _blend_residual_mean(frame: pd.DataFrame, *, stat: str) -> pd.DataFrame:
    """Shrink thin season-to-date toward last season / career, then apply residual."""

    season = pd.to_numeric(frame.get(f"season_avg_{stat}"), errors="coerce")
    trail = pd.to_numeric(frame.get(f"rolling_{stat}_8"), errors="coerce")
    career = pd.to_numeric(frame.get(f"career_{stat}"), errors="coerce")
    prior = pd.to_numeric(frame.get("prior_season_mean"), errors="coerce").fillna(career)
    season_n = pd.to_numeric(frame.get("season_games_before"), errors="coerce").fillna(0.0)
    shrunk = season.copy()
    have_prior = season.notna() & prior.notna()
    shrunk = shrunk.where(
        ~have_prior,
        (season_n * season + SEASON_PRIOR_STRENGTH * prior) / (season_n + SEASON_PRIOR_STRENGTH),
    )
    shrunk = shrunk.fillna(prior)
    early = season_n < 4
    weight = np.where(early, RECENCY_WEIGHT_EARLY, RECENCY_WEIGHT_STABLE)
    blended = shrunk.copy()
    both = shrunk.notna() & trail.notna()
    blended = blended.where(~both, (1.0 - weight) * shrunk + weight * trail)
    base = blended.fillna(shrunk).fillna(trail).fillna(career).fillna(MIN_MU)

    games = pd.to_numeric(frame.get("games_before"), errors="coerce").fillna(season_n)
    trail_targets = pd.to_numeric(frame.get("rolling_targets_8"), errors="coerce").fillna(0.0)
    n_window = games.clip(lower=1.0, upper=8.0)
    trail_target_mean = (trail_targets / n_window).replace(0.0, np.nan).fillna(4.0)
    expected = pd.to_numeric(frame.get("expected_targets"), errors="coerce").fillna(trail_target_mean)
    opp_ratio = (expected / trail_target_mean).clip(0.60, 1.50)
    volume = (
        pd.to_numeric(frame.get("expected_team_pass_attempts"), errors="coerce")
        / pd.to_numeric(frame.get("rolling_team_pass_attempts"), errors="coerce").replace(0.0, np.nan).fillna(32.0)
    ).clip(*VOLUME_CLIP)
    resid = 1.0 + OPP_RESIDUAL_WEIGHT * (opp_ratio - 1.0) + VOLUME_RESIDUAL_WEIGHT * (volume - 1.0)
    if "availability" in frame.columns:
        avail = pd.to_numeric(frame["availability"], errors="coerce").fillna(1.0).clip(AVAIL_OUT, 1.0)
    else:
        avail = 1.0
    out = frame.copy()
    out["season_games_before"] = season_n
    out["base_mu"] = base
    out["residual_scale"] = resid.clip(*RESIDUAL_CLIP)
    out["mu"] = (base * out["residual_scale"] * avail).clip(lower=MIN_MU)
    out["baseline_mu"] = season.fillna(career).fillna(trail).fillna(MIN_MU)
    out["trailing_median"] = trail
    return out


def apply_residual_mean(
    features: pd.DataFrame,
    *,
    stat: str = "receiving_yards",
) -> pd.DataFrame:
    """Season-to-date / trailing-8 mean with a shrunk opportunity residual."""

    return _blend_residual_mean(attach_stat_history(apply_opportunity_model(features), stat), stat=stat)


def attach_injury_availability(features: pd.DataFrame, injuries: pd.DataFrame | None) -> pd.DataFrame:
    """Join lagged weekly injury status. Missing reports stay fully available."""

    frame = features.copy()
    frame["availability"] = 1.0
    if injuries is None or injuries.empty:
        return frame
    inj = injuries.copy()
    if "player_id" not in inj.columns and "gsis_id" in inj.columns:
        inj["player_id"] = inj["gsis_id"]
    status_col = next((c for c in ("report_status", "injury_status", "status") if c in inj.columns), None)
    if status_col is None or "player_id" not in inj.columns:
        return frame
    keep = [c for c in ("season", "week", "player_id", status_col) if c in inj.columns]
    inj = inj[keep].dropna(subset=["player_id"])
    inj["availability"] = inj[status_col].map(injury_availability)
    inj = inj.sort_values([c for c in ("season", "week", "player_id") if c in inj.columns])
    inj = inj.drop_duplicates(["player_id", "season", "week"], keep="last")
    merged = frame.merge(
        inj[["player_id", "season", "week", "availability"]],
        on=["player_id", "season", "week"],
        how="left",
        suffixes=("", "_inj"),
    )
    if "availability_inj" in merged.columns:
        merged["availability"] = pd.to_numeric(merged["availability_inj"], errors="coerce").fillna(1.0)
        merged = merged.drop(columns=["availability_inj"])
    else:
        merged["availability"] = pd.to_numeric(merged["availability"], errors="coerce").fillna(1.0)
    return merged


def walk_forward_stat(
    features: pd.DataFrame,
    *,
    stat: str,
    eval_seasons: Iterable[int] = EVAL_SEASONS,
    synthetic_lines: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    modeled = apply_residual_mean(features, stat=stat)
    modeled = modeled[pd.to_numeric(modeled[stat], errors="coerce").notna()].copy()
    scored_parts = []
    alpha_by_season: dict[int, float] = {}
    for season in eval_seasons:
        prior = modeled[modeled["season"] < season]
        alpha = estimate_dispersion(prior[stat], prior["mu"]) if not prior.empty else DEFAULT_ALPHA
        alpha_by_season[int(season)] = float(alpha)
        current = modeled[modeled["season"].eq(season) & modeled["week"].between(1, 18)].copy()
        if current.empty:
            continue
        scored_parts.append(attach_distribution(current, alpha_by_season={int(season): alpha}))
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else modeled.iloc[0:0].copy()
    metrics = evaluate_stat(scored, stat=stat, synthetic_lines=synthetic_lines or YARD_LINES)
    metrics["alpha_by_season"] = alpha_by_season
    return scored, metrics


def evaluate_stat(
    scored: pd.DataFrame,
    *,
    stat: str,
    synthetic_lines: Sequence[float],
) -> dict[str, Any]:
    if scored.empty:
        return {"n": 0, "mae": None, "baseline_mae": None, "stat": stat}
    frame = scored.copy()
    frame["_actual"] = pd.to_numeric(frame[stat], errors="coerce")
    frame["mu"] = pd.to_numeric(frame["mu"], errors="coerce")
    frame["baseline_mu"] = pd.to_numeric(frame["baseline_mu"], errors="coerce")
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    frame = frame[frame["_actual"].notna() & frame["mu"].notna() & frame["baseline_mu"].notna()].copy()
    if frame.empty:
        return {"n": 0, "mae": None, "baseline_mae": None, "stat": stat}
    renamed = frame.copy()
    renamed["receiving_yards"] = frame["_actual"]
    renamed["median_yards"] = [
        predictive_quantile(mu, alpha, 0.5) for mu, alpha in zip(renamed["mu"], renamed["alpha"])
    ]
    from src.models import nfl_receiving_yards as yards_mod

    previous = yards_mod.SYNTHETIC_LINES
    yards_mod.SYNTHETIC_LINES = tuple(synthetic_lines)
    try:
        metrics = evaluate_predictions(renamed)
    finally:
        yards_mod.SYNTHETIC_LINES = previous
    metrics["stat"] = stat
    trail = pd.to_numeric(frame.get("trailing_median"), errors="coerce")
    actual = frame["_actual"].to_numpy(dtype=float)
    mu = frame["mu"].to_numpy(dtype=float)
    alpha = frame["alpha"].to_numpy(dtype=float)
    mask = trail.notna().to_numpy() & np.isfinite(trail.to_numpy(dtype=float))
    if mask.any():
        lines = trail.to_numpy(dtype=float)[mask]
        p = np.array([p_over_line(m, a, line) for m, a, line in zip(mu[mask], alpha[mask], lines)])
        empirical = (actual[mask] > lines).astype(float)
        metrics["trailing_median_calibration"] = {
            "n": int(mask.sum()),
            "mean_p_over": float(p.mean()),
            "empirical_over": float(empirical.mean()),
            "abs_gap": float(abs(p.mean() - empirical.mean())),
        }
    else:
        metrics["trailing_median_calibration"] = {"n": 0}
    return metrics


def advance_priors_for_next_game(frame: pd.DataFrame, *, stat: str) -> pd.DataFrame:
    """Roll the last played actual into season / trail / career for the next game."""

    out = frame.copy()
    actual = pd.to_numeric(out.get(stat), errors="coerce")
    season = pd.to_numeric(out.get(f"season_avg_{stat}"), errors="coerce")
    trail = pd.to_numeric(out.get(f"rolling_{stat}_8"), errors="coerce")
    career = pd.to_numeric(out.get(f"career_{stat}"), errors="coerce")
    season_n = pd.to_numeric(out.get("season_games_before"), errors="coerce").fillna(0.0)
    games = pd.to_numeric(out.get("games_before"), errors="coerce").fillna(0.0)
    played = actual.notna()
    new_season = np.where(
        ~played,
        season,
        np.where((season_n <= 0) | season.isna(), actual, (season * season_n + actual) / (season_n + 1.0)),
    )
    trail_n = season_n.clip(upper=8.0)
    new_trail = np.where(
        ~played,
        trail,
        np.where(
            (trail_n <= 0) | trail.isna(),
            actual,
            np.where(trail_n >= 8.0, (7.0 * trail + actual) / 8.0, (trail * trail_n + actual) / (trail_n + 1.0)),
        ),
    )
    new_career = np.where(
        ~played,
        career,
        np.where((games <= 0) | career.isna(), actual, (career * games + actual) / (games + 1.0)),
    )
    targets = pd.to_numeric(out.get("targets"), errors="coerce")
    trail_targets = pd.to_numeric(out.get("rolling_targets_8"), errors="coerce")
    if targets.notna().any():
        out["rolling_targets_8"] = np.where(
            targets.isna() | trail_targets.isna(),
            trail_targets,
            np.where(trail_n >= 8.0, trail_targets * 0.875 + targets, trail_targets + targets),
        )
    out[f"season_avg_{stat}"] = new_season
    out[f"rolling_{stat}_8"] = new_trail
    out[f"career_{stat}"] = new_career
    out["season_games_before"] = season_n + played.astype(float)
    out["games_before"] = games + played.astype(float)
    return out


def project_upcoming_rows(history: pd.DataFrame, upcoming: pd.DataFrame, *, stat: str = "receiving_yards") -> pd.DataFrame:
    """Copy each player's last modeled row onto upcoming team games.

    The last played actual is rolled into the prior first. Opponent, implied
    points, and availability are then refreshed.
    """

    if history.empty or upcoming.empty:
        return history.iloc[0:0].copy()
    latest = advance_priors_for_next_game(
        history.sort_values(["player_id", "game_date"]).groupby("player_id", sort=False).tail(1),
        stat=stat,
    )
    rows = []
    for game in upcoming.to_dict("records"):
        home = normalize_team_name(game.get("home_team"))
        away = normalize_team_name(game.get("away_team"))
        for team, opp, is_home in ((home, away, True), (away, home, False)):
            roster = latest[latest["team"] == team]
            for player in roster.to_dict("records"):
                row = dict(player)
                row["game_id"] = game.get("game_id")
                row["game_date"] = game.get("game_date") or game.get("gameday")
                row["season"] = game.get("season")
                row["week"] = game.get("week")
                row["home_team"] = home
                row["away_team"] = away
                row["opponent_team"] = opp
                row["is_home"] = 1.0 if is_home else 0.0
                row["predicted_margin"] = game.get("predicted_margin")
                row["closing_total"] = game.get("closing_total") or game.get("total_line")
                row["implied_team_points"] = implied_team_points(
                    row["predicted_margin"], row["closing_total"], is_home
                )
                row["receiving_yards"] = np.nan
                row["receptions"] = np.nan
                rows.append(row)
    projected = pd.DataFrame(rows)
    if projected.empty:
        return projected
    return refresh_live_mu(projected, stat=stat)


def refresh_live_mu(frame: pd.DataFrame, *, stat: str = "receiving_yards") -> pd.DataFrame:
    """Recompute residual mu after implied points / availability / prior change."""

    out = frame.copy()
    rolling_pass = pd.to_numeric(out.get("rolling_team_pass_attempts"), errors="coerce").fillna(32.0)
    rolling_points = pd.to_numeric(out.get("rolling_team_points"), errors="coerce").fillna(22.0).clip(lower=10.0)
    implied = pd.to_numeric(out.get("implied_team_points"), errors="coerce").fillna(rolling_points)
    volume = (implied / rolling_points).clip(*VOLUME_CLIP)
    share = pd.to_numeric(out.get("target_share_hat"), errors="coerce").fillna(0.10)
    out["expected_team_pass_attempts"] = rolling_pass * volume
    out["expected_targets"] = out["expected_team_pass_attempts"] * share
    return _blend_residual_mean(out, stat=stat)


def apply_name_availability(frame: pd.DataFrame, status_by_name: Mapping[str, str]) -> pd.DataFrame:
    out = frame.copy()
    names = out.get("player_display_name", out.get("player_name")).map(_norm_name)
    out["availability"] = names.map(lambda name: injury_availability(status_by_name.get(name, "")))
    return out


def consensus_sportsbook_line(payload: Mapping[str, Any], *, market: str) -> list[dict[str, Any]]:
    """One no-vig sportsbook line per player: modal point, best two-way prices."""

    compact = compact_closing_rows(payload, market=market)
    by_player: dict[str, list[dict[str, Any]]] = {}
    for row in compact:
        player = display_player_name(row.get("player"))
        if not player:
            continue
        row = dict(row)
        row["player"] = player
        by_player.setdefault(player, []).append(row)
    out = []
    for player, options in by_player.items():
        points = [row.get("point") for row in options]
        try:
            line = float(pd.Series(points).mode().iloc[0])
        except (IndexError, TypeError, ValueError):
            continue
        at_line = [row for row in options if float(row.get("point")) == line]
        def best_over(row: dict[str, Any]) -> float:
            price = row.get("over_price")
            implied = implied_prob_american(price)
            return implied if implied is not None else 9.0

        def best_under(row: dict[str, Any]) -> float:
            price = row.get("under_price")
            implied = implied_prob_american(price)
            return implied if implied is not None else 9.0

        over_row = min(at_line, key=best_over)
        under_row = min(at_line, key=best_under)
        novig = no_vig_over_prob(over_row.get("over_price"), under_row.get("under_price"))
        out.append(
            {
                "player": player,
                "market": market,
                "line": line,
                "best_over_book": over_row.get("book"),
                "best_under_book": under_row.get("book"),
                "best_over_price": over_row.get("over_price"),
                "best_under_price": under_row.get("under_price"),
                "book_novig_p_over": novig,
                "books": sorted({str(row.get("book")) for row in at_line}),
            }
        )
    return out


def _pick_projection(
    candidates: Sequence[Mapping[str, Any]],
    line: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    home = line.get("home_team")
    away = line.get("away_team")
    if home and away:
        on_event = [row for row in candidates if player_on_event(row.get("team"), home, away)]
    else:
        on_event = list(candidates)
    if not on_event:
        return None
    if len(on_event) == 1:
        return on_event[0]
    home_abbr = _team_abbr(home)
    away_abbr = _team_abbr(away)
    for row in on_event:
        team = _team_abbr(row.get("team"))
        expected_opp = away_abbr if team == home_abbr else home_abbr
        if _team_abbr(row.get("opponent_team")) == expected_opp:
            return row
    return on_event[0]


def score_live_lines(
    projections: pd.DataFrame,
    line_rows: Sequence[Mapping[str, Any]],
    *,
    stat: str,
) -> list[dict[str, Any]]:
    """Join live sportsbook lines to residual projections. No DFS, no EV/Kelly."""

    if projections.empty:
        return []
    proj = projections.copy()
    proj["_name"] = proj.get("player_display_name", proj.get("player_name")).map(_norm_name)
    lookup: dict[str, list[dict[str, Any]]] = {}
    for name, row in zip(proj["_name"], proj.to_dict("records")):
        if name:
            lookup.setdefault(name, []).append(row)
    scored = []
    seen: set[tuple[str, str, float]] = set()
    for line in line_rows:
        match = _pick_projection(lookup.get(_norm_name(line.get("player")), []), line)
        if not match:
            continue
        try:
            point = float(line.get("line"))
            mu = float(match["mu"])
            alpha = float(match.get("alpha") or DEFAULT_ALPHA)
        except (TypeError, ValueError):
            continue
        p_over = p_over_line(mu, alpha, point)
        team = match.get("team")
        opponent = match.get("opponent_team")
        if line.get("home_team") and line.get("away_team"):
            home = _team_abbr(line.get("home_team"))
            away = _team_abbr(line.get("away_team"))
            team_abbr = _team_abbr(team)
            if team_abbr not in {home, away}:
                continue
            opponent = away if team_abbr == home else home
        player = display_player_name(line.get("player"))
        key = (player, str(line.get("market") or stat), point)
        if key in seen:
            continue
        seen.add(key)
        book_p = line.get("book_novig_p_over")
        delta = None if book_p is None else float(p_over) - float(book_p)
        check_data = delta is not None and abs(delta) > CHECK_DATA_ABS_P
        scored.append(
            {
                "player": player,
                "team": team,
                "opponent": opponent,
                "position": match.get("position"),
                "market": line.get("market") or stat,
                "line": point,
                "our_mu": mu,
                "our_p_over": p_over,
                "our_fair_over": probability_to_american(p_over),
                "our_fair_under": probability_to_american(1.0 - p_over),
                "book_novig_p_over": book_p,
                "best_over_price": line.get("best_over_price"),
                "best_under_price": line.get("best_under_price"),
                "best_over_book": line.get("best_over_book"),
                "best_under_book": line.get("best_under_book"),
                "books": line.get("books"),
                "delta_p_over": delta,
                "status": "check_data" if check_data else "ok",
                "data_check": check_data,
                "season_games": match.get("season_games_before"),
                "season_avg": match.get(f"season_avg_{stat}"),
                "trailing_8": match.get(f"rolling_{stat}_8"),
                "prior_season_mean": match.get("prior_season_mean"),
                "residual_scale": match.get("residual_scale"),
                "availability": match.get("availability"),
                "alpha": alpha,
                "live_site_published": False,
            }
        )
    scored.sort(
        key=lambda row: (bool(row.get("data_check")), -abs(row.get("delta_p_over") or 0.0))
    )
    return scored


def reject_dfs_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clean = []
    for row in rows:
        books = [str(book).lower() for book in (row.get("books") or [])]
        if any(book in DFS_BOOK_KEYS for book in books):
            continue
        if classify_book(str(row.get("best_over_book") or "")) == "dfs":
            continue
        clean.append(dict(row))
    return clean
