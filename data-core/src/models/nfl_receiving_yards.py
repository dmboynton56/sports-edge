"""Walk-forward receiving-yards prototype (WR/TE/RB).

Opportunity (target share, snap/route proxy, team pass volume from a
spread/total) times efficiency (yards per target shrunk to position priors).
Outputs a negative-binomial distribution so P(over line) works for any line.

Leakage rule: every player/team feature is lagged with shift(1) before rolling.
This module does not load nflverse, score the live v2 artifact, or write serving
tables.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from src.models.nfl_v2 import normalize_team_name
from src.utils.team_codes import NFL_TEAM_NAME_TO_ABBR

TARGET_POSITIONS = ("WR", "TE", "RB")
YPT_PRIOR_STRENGTH = 24.0
TARGET_SHARE_PRIOR_STRENGTH = 8.0
SNAP_BLEND = 0.25
VOLUME_CLIP = (0.75, 1.35)
DEFAULT_ALPHA = 0.35
MIN_MU = 0.75
EVAL_SEASONS = (2023, 2024, 2025)
SYNTHETIC_LINES = (24.5, 49.5, 74.5)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _lagged_rolling(frame: pd.DataFrame, column: str, window: int, group: str = "player_id") -> pd.Series:
    return frame.groupby(group, sort=False)[column].transform(
        lambda values: values.shift(1).rolling(window, min_periods=1).mean()
    )


def _lagged_sum(frame: pd.DataFrame, column: str, window: int, group: str = "player_id") -> pd.Series:
    return frame.groupby(group, sort=False)[column].transform(
        lambda values: values.shift(1).rolling(window, min_periods=1).sum()
    )


def _lagged_expanding(frame: pd.DataFrame, column: str, group: str = "player_id") -> pd.Series:
    return frame.groupby(group, sort=False)[column].transform(
        lambda values: values.shift(1).expanding(min_periods=1).mean()
    )


def implied_team_points(predicted_margin: object, closing_total: object, is_home: bool) -> float | None:
    """Home margin + total → team implied points. Home points = (total + margin) / 2."""

    try:
        margin = float(predicted_margin)
        total = float(closing_total)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(margin) or not np.isfinite(total):
        return None
    home_points = (total + margin) / 2.0
    away_points = (total - margin) / 2.0
    points = home_points if is_home else away_points
    return float(points) if np.isfinite(points) else None


def nb_params(mu: float, alpha: float) -> tuple[float, float]:
    """Negative binomial with var = mu + alpha * mu^2 (alpha=0 → Poisson limit)."""

    mean = max(float(mu), MIN_MU)
    dispersion = max(float(alpha), 1e-6)
    n = 1.0 / dispersion
    p = 1.0 / (1.0 + dispersion * mean)
    return n, p


def p_over_line(mu: float, alpha: float, line: float) -> float:
    """P(yards > line). For a .5 line this is 1 - F(floor(line))."""

    n, p = nb_params(mu, alpha)
    threshold = int(np.floor(float(line)))
    return float(1.0 - nbinom.cdf(threshold, n, p))


def predictive_quantile(mu: float, alpha: float, tau: float) -> float:
    n, p = nb_params(mu, alpha)
    return float(nbinom.ppf(float(tau), n, p))


def crps_nb(actual: float, mu: float, alpha: float, k_max: int | None = None) -> float:
    """Discrete CRPS for a negative-binomial forecast of a count."""

    y = max(0, int(round(float(actual))))
    n, p = nb_params(mu, alpha)
    var = mu + max(alpha, 0.0) * mu * mu
    upper = int(k_max or max(y + 20, mu + 8.0 * np.sqrt(max(var, 1.0)), 40))
    ks = np.arange(0, upper + 1)
    forecast = nbinom.cdf(ks, n, p)
    observed = (ks >= y).astype(float)
    return float(np.sum((forecast - observed) ** 2))


def pinball_loss(actual: Sequence[float], predicted: Sequence[float], tau: float) -> float:
    y = np.asarray(actual, dtype=float)
    q = np.asarray(predicted, dtype=float)
    gap = y - q
    return float(np.mean(np.where(gap >= 0, tau * gap, (tau - 1.0) * gap)))


def estimate_dispersion(actual: Sequence[float], mu: Sequence[float]) -> float:
    y = np.asarray(actual, dtype=float)
    m = np.clip(np.asarray(mu, dtype=float), MIN_MU, None)
    mask = np.isfinite(y) & np.isfinite(m)
    if mask.sum() < 30:
        return DEFAULT_ALPHA
    sse = np.mean((y[mask] - m[mask]) ** 2)
    mean_mu = float(np.mean(m[mask]))
    mean_mu_sq = float(np.mean(m[mask] ** 2))
    alpha = (sse - mean_mu) / max(mean_mu_sq, 1e-6)
    return float(np.clip(alpha, 0.05, 1.5))


def _prepare_player_games(player_games: pd.DataFrame) -> pd.DataFrame:
    stats = player_games.copy()
    if "team" not in stats.columns and "recent_team" in stats.columns:
        stats["team"] = stats["recent_team"]
    if "player_display_name" not in stats.columns:
        stats["player_display_name"] = stats.get("player_name")
    stats["position"] = stats["position"].astype(str).str.upper()
    stats = stats[stats["position"].isin(TARGET_POSITIONS)].copy()
    stats["team"] = stats["team"].map(normalize_team_name)
    if "opponent_team" in stats.columns:
        stats["opponent_team"] = stats["opponent_team"].map(normalize_team_name)
    stats["targets"] = _num(stats, "targets")
    stats["receiving_yards"] = _num(stats, "receiving_yards")
    stats["receptions"] = _num(stats, "receptions")
    stats["attempts"] = _num(stats, "attempts")
    stats["carries"] = _num(stats, "carries")
    stats["offense_pct"] = _num(stats, "offense_pct", default=np.nan)
    stats["target_share"] = _num(stats, "target_share", default=np.nan)
    stats["ypt"] = np.where(stats["targets"] > 0, stats["receiving_yards"] / stats["targets"], np.nan)
    participated = stats["targets"].gt(0) | stats["receptions"].gt(0) | stats["offense_pct"].fillna(0).gt(0)
    return stats[participated].copy()


def _attach_schedule(stats: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    games = schedule.copy()
    if "game_date" not in games.columns:
        games["game_date"] = games.get("gameday")
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["home_team"] = games["home_team"].map(normalize_team_name)
    games["away_team"] = games["away_team"].map(normalize_team_name)
    if "predicted_margin" not in games.columns:
        games["predicted_margin"] = np.nan
    if "closing_total" not in games.columns:
        games["closing_total"] = pd.to_numeric(games.get("total_line"), errors="coerce")
    keep = [
        column
        for column in (
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "predicted_margin",
            "closing_total",
            "spread_source",
            "total_source",
        )
        if column in games.columns
    ]
    games = games[keep].drop_duplicates("game_id", keep="last")
    merged = stats.merge(games, on="game_id", how="left", suffixes=("", "_schedule"))
    merged["is_home"] = (merged["team"] == merged["home_team"]).astype(float)
    return merged.sort_values(["game_date", "season", "week", "player_id"]).reset_index(drop=True)


def _team_volume(player_games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Team pass attempts from all positions (QB attempts), not just WR/TE/RB."""

    raw = player_games.copy()
    if "team" not in raw.columns and "recent_team" in raw.columns:
        raw["team"] = raw["recent_team"]
    raw["team"] = raw["team"].map(normalize_team_name)
    raw["attempts"] = _num(raw, "attempts")
    raw["carries"] = _num(raw, "carries")
    raw["targets"] = _num(raw, "targets")
    volume = raw.groupby(["game_id", "team"], dropna=False, as_index=False).agg(
        team_pass_attempts=("attempts", "sum"),
        team_carries=("carries", "sum"),
        team_targets=("targets", "sum"),
    )
    games = schedule.copy()
    if "game_date" not in games.columns:
        games["game_date"] = games.get("gameday")
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    volume = volume.merge(games[["game_id", "game_date"]].drop_duplicates("game_id"), on="game_id", how="left")
    volume = volume.sort_values(["team", "game_date", "game_id"])
    volume["rolling_team_pass_attempts"] = _lagged_rolling(volume, "team_pass_attempts", 5, group="team")
    volume["rolling_team_plays"] = volume.groupby("team", sort=False)["team_pass_attempts"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    ) + volume.groupby("team", sort=False)["team_carries"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )
    return volume


def build_feature_frame(
    player_games: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Leakage-safe receiving-yards features, one row per player-game."""

    stats = _attach_schedule(_prepare_player_games(player_games), schedule)
    team_games = _team_volume(player_games, schedule)
    team_games["rolling_team_pass_attempts"] = _lagged_rolling(team_games, "team_pass_attempts", 5, group="team")
    team_games["rolling_team_plays"] = team_games.groupby("team", sort=False)["team_pass_attempts"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    ) + team_games.groupby("team", sort=False)["team_carries"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )
    stats = stats.merge(
        team_games[["game_id", "team", "team_pass_attempts", "rolling_team_pass_attempts", "rolling_team_plays"]],
        on=["game_id", "team"],
        how="left",
    )
    missing_share = stats["target_share"].isna() & stats["team_pass_attempts"].gt(0)
    stats.loc[missing_share, "target_share"] = (
        stats.loc[missing_share, "targets"] / stats.loc[missing_share, "team_pass_attempts"]
    )

    stats["rolling_targets_8"] = _lagged_sum(stats, "targets", 8)
    stats["rolling_ypt_8"] = _lagged_rolling(stats, "ypt", 8)
    stats["rolling_target_share_5"] = _lagged_rolling(stats, "target_share", 5)
    stats["rolling_yards_5"] = _lagged_rolling(stats, "receiving_yards", 5)
    stats["rolling_snap_pct_5"] = _lagged_rolling(stats, "offense_pct", 5)
    stats["career_ypt"] = _lagged_expanding(stats, "ypt")
    stats["career_target_share"] = _lagged_expanding(stats, "target_share")
    stats["career_yards"] = _lagged_expanding(stats, "receiving_yards")
    stats["games_before"] = stats.groupby("player_id", sort=False).cumcount().astype(float)

    season_shift = stats.groupby(["player_id", "season"], sort=False)["receiving_yards"].shift(1)
    stats["season_avg_yards"] = season_shift.groupby([stats["player_id"], stats["season"]]).transform(
        lambda values: values.expanding(min_periods=1).mean()
    )
    stats["implied_team_points"] = [
        implied_team_points(margin, total, bool(is_home))
        for margin, total, is_home in zip(stats["predicted_margin"], stats["closing_total"], stats["is_home"])
    ]
    stats["implied_team_points"] = pd.to_numeric(stats["implied_team_points"], errors="coerce")
    team_points = np.where(
        stats["is_home"].eq(1),
        pd.to_numeric(stats.get("home_score"), errors="coerce"),
        pd.to_numeric(stats.get("away_score"), errors="coerce"),
    )
    if "home_score" in stats.columns:
        stats["_team_points_actual"] = team_points
        stats["rolling_team_points"] = _lagged_rolling(stats, "_team_points_actual", 5, group="team")
    else:
        stats["rolling_team_points"] = np.nan
    return stats


def apply_opportunity_model(features: pd.DataFrame) -> pd.DataFrame:
    """Attach shrunk opportunity/efficiency mean and NB parameters."""

    frame = features.copy()
    # Position priors are lagged expanding means, not full-sample stats.
    position_ypt = frame.groupby("position", sort=False)["ypt"].transform(
        lambda values: values.shift(1).expanding(min_periods=8).mean()
    )
    position_share = frame.groupby("position", sort=False)["target_share"].transform(
        lambda values: values.shift(1).expanding(min_periods=8).mean()
    )
    fallback_ypt = frame["position"].map({"WR": 8.2, "TE": 7.4, "RB": 6.4})
    fallback_share = frame["position"].map({"WR": 0.12, "TE": 0.08, "RB": 0.06})
    pos_ypt = 0.5 * position_ypt.fillna(fallback_ypt) + 0.5 * fallback_ypt
    pos_share = 0.5 * position_share.fillna(fallback_share) + 0.5 * fallback_share

    n_targets = frame["rolling_targets_8"].fillna(0.0)
    raw_ypt = frame["rolling_ypt_8"].fillna(frame["career_ypt"]).fillna(pos_ypt)
    frame["ypt_hat"] = (n_targets * raw_ypt + YPT_PRIOR_STRENGTH * pos_ypt) / (n_targets + YPT_PRIOR_STRENGTH)

    n_games = frame["games_before"].clip(lower=0)
    raw_share = frame["rolling_target_share_5"].fillna(frame["career_target_share"]).fillna(pos_share)
    share_hat = (n_games * raw_share + TARGET_SHARE_PRIOR_STRENGTH * pos_share) / (
        n_games + TARGET_SHARE_PRIOR_STRENGTH
    )
    snap = frame["rolling_snap_pct_5"]
    frame["target_share_hat"] = np.where(
        snap.notna(),
        (1.0 - SNAP_BLEND) * share_hat + SNAP_BLEND * snap.clip(0.0, 1.0) * pos_share / 0.55,
        share_hat,
    )
    frame["target_share_hat"] = frame["target_share_hat"].clip(0.01, 0.45)

    rolling_pass = frame["rolling_team_pass_attempts"].fillna(32.0)
    rolling_points = frame["rolling_team_points"].fillna(22.0).clip(lower=10.0)
    implied = frame["implied_team_points"].fillna(rolling_points)
    volume_scale = (implied / rolling_points).clip(*VOLUME_CLIP)
    frame["expected_team_pass_attempts"] = rolling_pass * volume_scale
    frame["expected_targets"] = frame["expected_team_pass_attempts"] * frame["target_share_hat"]
    frame["mu"] = (frame["expected_targets"] * frame["ypt_hat"]).clip(lower=MIN_MU)
    frame["baseline_mu"] = (
        frame["season_avg_yards"].fillna(frame["career_yards"]).fillna(frame["rolling_yards_5"]).fillna(pos_ypt * 4.0)
    )
    return frame


def attach_distribution(frame: pd.DataFrame, *, alpha_by_season: Mapping[int, float] | None = None) -> pd.DataFrame:
    out = frame.copy()
    default_alpha = DEFAULT_ALPHA
    alphas = []
    for season in out["season"].tolist():
        if alpha_by_season and int(season) in alpha_by_season:
            alphas.append(float(alpha_by_season[int(season)]))
        else:
            alphas.append(default_alpha)
    out["alpha"] = alphas
    out["median_yards"] = [predictive_quantile(mu, alpha, 0.5) for mu, alpha in zip(out["mu"], out["alpha"])]
    out["p_over_median"] = [
        p_over_line(mu, alpha, median) for mu, alpha, median in zip(out["mu"], out["alpha"], out["median_yards"])
    ]
    return out


def walk_forward_evaluate(features: pd.DataFrame, *, eval_seasons: Iterable[int] = EVAL_SEASONS) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score 2023–2025 with dispersion fit on prior seasons only."""

    scored_parts = []
    alpha_by_season: dict[int, float] = {}
    modeled = apply_opportunity_model(features)
    modeled = modeled[modeled["receiving_yards"].notna()].copy()
    for season in eval_seasons:
        prior = modeled[modeled["season"] < season]
        alpha = estimate_dispersion(prior["receiving_yards"], prior["mu"]) if not prior.empty else DEFAULT_ALPHA
        alpha_by_season[int(season)] = alpha
        current = modeled[modeled["season"].eq(season) & modeled["week"].between(1, 18)].copy()
        if current.empty:
            continue
        scored_parts.append(attach_distribution(current, alpha_by_season={int(season): alpha}))
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else modeled.iloc[0:0].copy()
    metrics = evaluate_predictions(scored)
    usage = scored[
        (scored["rolling_targets_8"].fillna(0) >= 8)
        | (scored["rolling_yards_5"].fillna(0) >= 20)
        | (scored["season_avg_yards"].fillna(0) >= 25)
    ]
    metrics["board_like"] = evaluate_predictions(usage)
    metrics["alpha_by_season"] = alpha_by_season
    return scored, metrics


def _calibration_table(actual: np.ndarray, probability: np.ndarray, bins: int = 8) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1.0 else probability <= upper)
        if not mask.any():
            continue
        rows.append(
            {
                "lo": float(lower),
                "hi": float(upper),
                "n": int(mask.sum()),
                "mean_p": float(probability[mask].mean()),
                "empirical": float(actual[mask].mean()),
            }
        )
    return rows


def evaluate_predictions(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty:
        return {"n": 0, "mae": None, "baseline_mae": None}
    frame = scored.copy()
    frame["receiving_yards"] = pd.to_numeric(frame["receiving_yards"], errors="coerce")
    frame["mu"] = pd.to_numeric(frame["mu"], errors="coerce")
    frame["baseline_mu"] = pd.to_numeric(frame["baseline_mu"], errors="coerce")
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    frame["median_yards"] = pd.to_numeric(frame["median_yards"], errors="coerce")
    frame = frame[frame["receiving_yards"].notna() & frame["mu"].notna() & frame["baseline_mu"].notna()].copy()
    if frame.empty:
        return {"n": 0, "mae": None, "baseline_mae": None}
    y = frame["receiving_yards"].to_numpy(dtype=float)
    mu = frame["mu"].to_numpy(dtype=float)
    baseline = frame["baseline_mu"].to_numpy(dtype=float)
    alpha = frame["alpha"].to_numpy(dtype=float)
    median = frame["median_yards"].to_numpy(dtype=float)
    over_median = (y > median).astype(float)
    p_over_median = np.array(
        [p_over_line(m, a, med) for m, a, med in zip(mu, alpha, median)], dtype=float
    )
    crps = np.array([crps_nb(actual, m, a) for actual, m, a in zip(y, mu, alpha)], dtype=float)
    q10 = np.array([predictive_quantile(m, a, 0.1) for m, a in zip(mu, alpha)])
    q50 = np.array([predictive_quantile(m, a, 0.5) for m, a in zip(mu, alpha)])
    q90 = np.array([predictive_quantile(m, a, 0.9) for m, a in zip(mu, alpha)])
    line_rows = []
    for line in SYNTHETIC_LINES:
        p = np.array([p_over_line(m, a, line) for m, a in zip(mu, alpha)])
        empirical = (y > line).astype(float)
        line_rows.append(
            {
                "line": line,
                "n": int(len(y)),
                "mean_p_over": float(p.mean()),
                "empirical_over": float(empirical.mean()),
                "abs_gap": float(abs(p.mean() - empirical.mean())),
                "calibration": _calibration_table(empirical, p),
            }
        )
    by_season = []
    seasons = frame["season"] if "season" in frame.columns else pd.Series([None] * len(y), index=frame.index)
    positions = frame["position"] if "position" in frame.columns else pd.Series(["NA"] * len(y), index=frame.index)
    for season in sorted(pd.unique(pd.to_numeric(seasons, errors="coerce").dropna().astype(int))):
        mask = pd.to_numeric(seasons, errors="coerce").eq(season).to_numpy()
        by_season.append(
            {
                "season": int(season),
                "n": int(mask.sum()),
                "mae": float(np.mean(np.abs(y[mask] - mu[mask]))),
                "baseline_mae": float(np.mean(np.abs(y[mask] - baseline[mask]))),
                "crps": float(np.mean(crps[mask])),
            }
        )
    by_position = []
    for position in TARGET_POSITIONS:
        mask = positions.astype(str).eq(position).to_numpy()
        if not mask.any():
            continue
        by_position.append(
            {
                "position": position,
                "n": int(mask.sum()),
                "mae": float(np.mean(np.abs(y[mask] - mu[mask]))),
                "baseline_mae": float(np.mean(np.abs(y[mask] - baseline[mask]))),
            }
        )
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - mu))),
        "baseline_mae": float(np.mean(np.abs(y - baseline))),
        "mae_lift": float(np.mean(np.abs(y - baseline)) - np.mean(np.abs(y - mu))),
        "crps": float(np.mean(crps)),
        "pinball_10": pinball_loss(y, q10, 0.1),
        "pinball_50": pinball_loss(y, q50, 0.5),
        "pinball_90": pinball_loss(y, q90, 0.9),
        "median_over_rate": float(over_median.mean()),
        "median_p_over_mean": float(p_over_median.mean()),
        "median_calibration": _calibration_table(over_median, p_over_median),
        "synthetic_lines": line_rows,
        "by_season": by_season,
        "by_position": by_position,
        "roi_clv": {
            "status": "not_run",
            "reason": "No historical receiving-yard closing lines were sampled (Odds API historical is 10 credits/event/market; a 1-week close sample is not cheap against the shared Starter budget).",
            "n": 0,
        },
    }


def _norm_player(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()


def grade_closing_sample(scored: pd.DataFrame, sample: Mapping[str, Any]) -> dict[str, Any]:
    """Tiny ROI readout for a one-game PropLine closing sample. Small-sample only."""

    rows = list(sample.get("closing_sample") or [])
    if not rows or scored.empty:
        return {
            "status": "not_run",
            "reason": "No closing-line sample to join.",
            "n": 0,
            "label": "small-sample",
        }
    home = NFL_TEAM_NAME_TO_ABBR.get(str(sample.get("home_team") or "").lower(), normalize_team_name(sample.get("home_team")))
    away = NFL_TEAM_NAME_TO_ABBR.get(str(sample.get("away_team") or "").lower(), normalize_team_name(sample.get("away_team")))
    games = scored.copy()
    if home and away and "home_team" in games.columns:
        matched = games[
            ((games["home_team"] == home) & (games["away_team"] == away))
            | ((games["home_team"] == away) & (games["away_team"] == home))
        ]
        if not matched.empty:
            games = matched
    games["_name"] = games.get("player_display_name", games.get("player_name")).map(_norm_player)
    name_to_row = {name: row for name, row in zip(games["_name"], games.to_dict("records")) if name}
    bets = []
    for line in rows:
        player = _norm_player(line.get("player"))
        match = name_to_row.get(player)
        if not match:
            continue
        try:
            point = float(line.get("point"))
            over_price = float(line.get("over_price"))
        except (TypeError, ValueError):
            continue
        p_over = p_over_line(float(match["mu"]), float(match["alpha"]), point)
        actual = float(match["receiving_yards"])
        side = "over" if p_over >= 0.5 else "under"
        won = actual > point if side == "over" else actual < point
        pushed = actual == point
        price = over_price if side == "over" else float(line.get("under_price") or -110)
        if pushed:
            profit = 0.0
        elif won:
            profit = price / 100.0 if price > 0 else 100.0 / abs(price)
        else:
            profit = -1.0
        bets.append(
            {
                "player": line.get("player"),
                "book": line.get("book"),
                "line": point,
                "side": side,
                "p_over": p_over,
                "actual": actual,
                "profit": profit,
            }
        )
    if not bets:
        return {
            "status": "sampled_unjoined",
            "reason": "Closing sample did not join to prototype rows (name/date/team).",
            "n": 0,
            "label": "small-sample",
            "sample_event": {
                "home_team": sample.get("home_team"),
                "away_team": sample.get("away_team"),
                "commence_time": sample.get("commence_time"),
                "rows": len(rows),
            },
        }
    profits = [bet["profit"] for bet in bets]
    return {
        "status": "small_sample",
        "label": "small-sample — not a strategy, not CLV vs open, one event",
        "n": len(bets),
        "unique_players": len({bet["player"] for bet in bets}),
        "books": sorted({bet["book"] for bet in bets}),
        "flat_roi": float(sum(profits) / len(profits)),
        "units": float(sum(profits)),
        "win": int(sum(1 for bet in bets if bet["profit"] > 0)),
        "loss": int(sum(1 for bet in bets if bet["profit"] < 0)),
        "push": int(sum(1 for bet in bets if bet["profit"] == 0)),
        "sample_event": {
            "home_team": sample.get("home_team"),
            "away_team": sample.get("away_team"),
            "commence_time": sample.get("commence_time"),
        },
    }
