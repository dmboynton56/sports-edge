"""
Rebuild notebooks/cache/pga_feature_store_event_level.csv from archived TSV + weather.

Uses a LEFT join on weather (notebook used inner, which dropped post-2021 events).
Missing Course Par -> 72; missing wind speeds -> 10 mph so wind features stay defined.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ARCHIVE_DIR = Path(__file__).resolve().parent / "archive"
DEFAULT_RESULTS = ARCHIVE_DIR / "pga_results_2001-2025.tsv"
DEFAULT_RESULTS_SUPPLEMENT = ARCHIVE_DIR / "pga_results_espn_supplement.tsv"
DEFAULT_WEATHER = ARCHIVE_DIR / "Tournaments With Weather Data.csv"
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent.parent
    / "notebooks"
    / "cache"
    / "pga_feature_store_event_level.csv"
)

# Majors + PLAYERS — used for SG weighting and "strong field" schedule context
STRONG_FIELD_EVENTS = frozenset(
    {
        "masters tournament",
        "pga championship",
        "u.s. open",
        "the open",
        "the open championship",
        "the players championship",
    }
)

# Continuous field-strength SG scaling parameters
FIELD_STRENGTH_ALPHA = 0.5          # sensitivity: how much field quality shifts the scale factor
FIELD_STRENGTH_SCALE_MIN = 0.80     # floor: weakest fields get at most a 20% SG haircut
FIELD_STRENGTH_SCALE_MAX = 1.15     # ceiling: strongest fields get at most a 15% SG boost
LIV_STRUCTURAL_DISCOUNT = 0.95      # small no-cut / 3-round format discount applied on top


def _apply_continuous_field_scaling(
    df: pd.DataFrame,
    alpha: float = FIELD_STRENGTH_ALPHA,
    scale_min: float = FIELD_STRENGTH_SCALE_MIN,
    scale_max: float = FIELD_STRENGTH_SCALE_MAX,
    liv_structural_discount: float = LIV_STRUCTURAL_DISCOUNT,
) -> pd.DataFrame:
    """Replace binary SG multipliers with a continuous field-strength factor.

    1. Compute each player's prior career-average SG from raw (unscaled) values.
    2. Per-event field quality = mean of those career SGs across all entrants.
    3. Scale = clip(1 + alpha * (field_quality - global_median), min, max).
    4. LIV events get an additional structural discount for the no-cut format.
    """
    sg_cols = ["sg_total_r1", "sg_total_r2", "sg_total_r3", "sg_total_r4"]

    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df = df.sort_values(["name", "start", "tournament"]).reset_index(drop=True)

    df["_raw_sg_avg"] = df[sg_cols].mean(axis=1)
    df["_career_sg"] = _group_shift_expanding_mean(df, ["name"], "_raw_sg_avg")

    df["_fs"] = df.groupby(["season", "tournament"], sort=False)["_career_sg"].transform(
        "mean"
    )
    ref = df["_fs"].median()
    df["_scale"] = np.clip(1.0 + alpha * (df["_fs"] - ref), scale_min, scale_max)
    df["_scale"] = df["_scale"].fillna(1.0)

    is_liv = df["tournament_lower"].str.contains(r"\bliv golf\b", case=False, na=False)
    df.loc[is_liv, "_scale"] *= liv_structural_discount

    for col in sg_cols:
        df[col] *= df["_scale"]
    df["sg_total_tournament"] = df[sg_cols].sum(axis=1)

    df = df.drop(columns=["_raw_sg_avg", "_career_sg", "_fs", "_scale"])
    return df


def load_base_frames(
    results_path: Path = DEFAULT_RESULTS,
    weather_path: Path = DEFAULT_WEATHER,
    weather_join: str = "left",
    supplement_path: Optional[Path] = None,
    use_supplement: bool = True,
) -> pd.DataFrame:
    df_results = pd.read_csv(results_path, sep="\t")
    sup_path = supplement_path
    if use_supplement and sup_path is None:
        sup_path = DEFAULT_RESULTS_SUPPLEMENT
    if use_supplement and sup_path and Path(sup_path).exists():
        df_sup = pd.read_csv(sup_path, sep="\t")
        # Align columns (ignore extras in supplement)
        for col in df_results.columns:
            if col not in df_sup.columns:
                df_sup[col] = ""
        df_sup = df_sup[[c for c in df_results.columns if c in df_sup.columns]]
        df_results = pd.concat([df_results, df_sup], ignore_index=True)
        df_results = df_results.drop_duplicates(
            subset=["season", "start", "tournament", "name"], keep="last"
        )
    df_weather = pd.read_csv(weather_path)

    for r in ["round1", "round2", "round3", "round4"]:
        df_results[r] = pd.to_numeric(df_results[r], errors="coerce")
        # A valid golf round is never 0 strokes; ESPN returns 0 for WD/DNS entries.
        df_results.loc[df_results[r] == 0, r] = np.nan
    df_results["total"] = pd.to_numeric(df_results["total"], errors="coerce")

    df_weather["season"] = df_weather["year"]
    df_weather["tournament_lower"] = df_weather["name"].str.lower().str.strip()
    df_results["tournament_lower"] = df_results["tournament"].str.lower().str.strip()

    how = "inner" if weather_join.strip().lower() == "inner" else "left"
    df = pd.merge(
        df_results,
        df_weather[
            ["season", "tournament_lower", "Course Par", "day0wind", "day1wind", "day2wind"]
        ],
        on=["season", "tournament_lower"],
        how=how,
    )
    df["Course Par"] = pd.to_numeric(df["Course Par"], errors="coerce").fillna(72.0)
    for wc in ["day0wind", "day1wind", "day2wind"]:
        df[wc] = pd.to_numeric(df[wc], errors="coerce").fillna(10.0)

    df["r1_vs_par"] = df["round1"] - df["Course Par"]
    df["r2_vs_par"] = df["round2"] - df["Course Par"]
    df["r3_vs_par"] = df["round3"] - df["Course Par"]
    df["r4_vs_par"] = df["round4"] - df["Course Par"]

    gcols = ["season", "tournament"]
    df["field_avg_r1"] = df.groupby(gcols)["r1_vs_par"].transform("mean")
    df["field_avg_r2"] = df.groupby(gcols)["r2_vs_par"].transform("mean")
    df["field_avg_r3"] = df.groupby(gcols)["r3_vs_par"].transform("mean")
    df["field_avg_r4"] = df.groupby(gcols)["r4_vs_par"].transform("mean")

    df["sg_total_r1"] = df["field_avg_r1"] - df["r1_vs_par"]
    df["sg_total_r2"] = df["field_avg_r2"] - df["r2_vs_par"]
    df["sg_total_r3"] = df["field_avg_r3"] - df["r3_vs_par"]
    df["sg_total_r4"] = df["field_avg_r4"] - df["r4_vs_par"]
    df["sg_total_tournament"] = df[
        ["sg_total_r1", "sg_total_r2", "sg_total_r3", "sg_total_r4"]
    ].sum(axis=1)

    df = _apply_continuous_field_scaling(df)
    return df


def _group_shift_expanding_mean(
    df: pd.DataFrame, group_cols: list[str], value_col: str
) -> pd.Series:
    """Per-group mean of prior rows only (same as shift().expanding().mean()); no Python lambdas."""
    sh = df.groupby(group_cols, sort=False)[value_col].shift()
    keys = [df[c] for c in group_cols]
    sh_num = pd.to_numeric(sh, errors="coerce")
    num = sh_num.groupby(keys, sort=False).cumsum()
    den = sh.notna().astype(np.int64).groupby(keys, sort=False).cumsum()
    return (num / den.astype(float)).where(den > 0)


def _group_shift_rolling_mean(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    window: int,
    min_periods: int,
) -> pd.Series:
    sh = df.groupby(group_cols, sort=False)[value_col].shift()
    keys = [df[c] for c in group_cols]
    r = sh.groupby(keys, sort=False).rolling(window, min_periods=min_periods).mean()
    return r.reset_index(level=list(range(len(group_cols))), drop=True)


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    df_model = df.copy()
    df_model["start"] = pd.to_datetime(df_model["start"], errors="coerce")
    df_model["end"] = pd.to_datetime(df_model["end"], errors="coerce")
    df_model["position_str"] = df_model["position"].astype(str).str.strip()
    df_model["position_num"] = pd.to_numeric(
        df_model["position_str"].str.extract(r"(\d+)")[0], errors="coerce"
    )

    df_model["is_win"] = (df_model["position_str"] == "1").astype(int)
    df_model["is_top5"] = (df_model["position_num"] <= 5).fillna(False).astype(int)
    df_model["is_top10"] = (df_model["position_num"] <= 10).fillna(False).astype(int)
    df_model["is_top20"] = (df_model["position_num"] <= 20).fillna(False).astype(int)
    df_model["made_cut"] = (df_model["round3"].notna() | df_model["round4"].notna()).astype(
        int
    )

    df_model["rounds_played"] = df_model[["round1", "round2", "round3", "round4"]].notna().sum(
        axis=1
    )
    df_model["sg_avg_round"] = df_model[
        ["sg_total_r1", "sg_total_r2", "sg_total_r3", "sg_total_r4"]
    ].mean(axis=1)
    df_model["sg_front_half"] = df_model[["sg_total_r1", "sg_total_r2"]].mean(axis=1)
    df_model["sg_back_half"] = df_model[["sg_total_r3", "sg_total_r4"]].mean(axis=1)
    df_model["sg_close_delta"] = df_model["sg_back_half"] - df_model["sg_front_half"]
    df_model["round_score_std"] = df_model[
        ["r1_vs_par", "r2_vs_par", "r3_vs_par", "r4_vs_par"]
    ].std(axis=1)

    df_model = df_model.sort_values(["name", "start", "tournament"]).reset_index(drop=True)
    player_cols: list[str] = ["name"]
    event_cols: list[str] = ["name", "tournament"]
    player_group = df_model.groupby("name", sort=False)

    df_model["starts_before"] = player_group.cumcount()
    df_model["prev_avg_sg_total"] = _group_shift_expanding_mean(
        df_model, player_cols, "sg_total_tournament"
    )
    df_model["prev_avg_sg_round"] = _group_shift_expanding_mean(
        df_model, player_cols, "sg_avg_round"
    )
    df_model["prev_avg_rounds_played"] = _group_shift_expanding_mean(
        df_model, player_cols, "rounds_played"
    )
    df_model["prev_cut_rate"] = _group_shift_expanding_mean(df_model, player_cols, "made_cut")
    df_model["prev_win_rate"] = _group_shift_expanding_mean(df_model, player_cols, "is_win")
    df_model["prev_top5_rate"] = _group_shift_expanding_mean(df_model, player_cols, "is_top5")
    df_model["prev_top10_rate"] = _group_shift_expanding_mean(df_model, player_cols, "is_top10")
    df_model["prev_top20_rate"] = _group_shift_expanding_mean(df_model, player_cols, "is_top20")
    df_model["prev_avg_finish_num"] = _group_shift_expanding_mean(
        df_model, player_cols, "position_num"
    )

    df_model["prev_avg_r1_sg"] = _group_shift_expanding_mean(
        df_model, player_cols, "sg_total_r1"
    )
    df_model["prev_avg_r4_sg"] = _group_shift_expanding_mean(
        df_model, player_cols, "sg_total_r4"
    )
    df_model["prev_avg_close_delta"] = _group_shift_expanding_mean(
        df_model, player_cols, "sg_close_delta"
    )

    for window in [3, 5, 10, 20]:
        min_periods = 2 if window <= 5 else 5
        df_model[f"prev_sg_form_{window}"] = _group_shift_rolling_mean(
            df_model, player_cols, "sg_avg_round", window, min_periods
        )

    df_model["prev_form_trend_5v20"] = (
        df_model["prev_sg_form_5"] - df_model["prev_sg_form_20"]
    )
    df_model["prev_round_std_10"] = _group_shift_rolling_mean(
        df_model, player_cols, "round_score_std", 10, 5
    )

    event_group = df_model.groupby(event_cols, sort=False)
    df_model["event_starts_before"] = event_group.cumcount()
    df_model["prev_event_avg_sg_round"] = _group_shift_expanding_mean(
        df_model, event_cols, "sg_avg_round"
    )
    df_model["prev_event_cut_rate"] = _group_shift_expanding_mean(
        df_model, event_cols, "made_cut"
    )
    df_model["prev_event_top20_rate"] = _group_shift_expanding_mean(
        df_model, event_cols, "is_top20"
    )

    # When event_starts_before == 0 (new venue), fall back to player career avgs
    # instead of leaving NaN (which would later be filled with PGA-centric train medians).
    new_venue = df_model["event_starts_before"] == 0
    fallback_map = {
        "prev_event_avg_sg_round": "prev_avg_sg_round",
        "prev_event_cut_rate": "prev_cut_rate",
        "prev_event_top20_rate": "prev_top20_rate",
    }
    for event_col, career_col in fallback_map.items():
        df_model.loc[new_venue & df_model[event_col].isna(), event_col] = (
            df_model.loc[new_venue & df_model[event_col].isna(), career_col]
        )

    g_event = df_model.groupby(["season", "tournament"], sort=False)
    df_model["field_strength_prev_avg_sg"] = g_event["prev_avg_sg_round"].transform("mean")
    df_model["field_strength_median_prev_sg"] = g_event["prev_avg_sg_round"].transform("median")

    df_model["relative_skill_vs_field"] = (
        df_model["prev_avg_sg_round"] - df_model["field_strength_prev_avg_sg"]
    )
    df_model["relative_skill_vs_field_median"] = (
        df_model["prev_avg_sg_round"] - df_model["field_strength_median_prev_sg"]
    )

    # Field size — lets models distinguish small LIV fields (48-54)
    # from full PGA fields (120-156).
    df_model["field_size"] = g_event["name"].transform("count")

    tl = df_model["tournament"].str.lower().str.strip()
    df_model["is_liv_event"] = tl.str.contains(r"\bliv golf\b", case=False, na=False).astype(
        np.float64
    )
    df_model["is_strong_field_event"] = tl.isin(STRONG_FIELD_EVENTS).astype(np.float64)
    for win, min_p in [(10, 3), (20, 5)]:
        df_model[f"liv_share_last_{win}"] = _group_shift_rolling_mean(
            df_model, player_cols, "is_liv_event", win, min_p
        )
        df_model[f"strong_field_share_last_{win}"] = _group_shift_rolling_mean(
            df_model, player_cols, "is_strong_field_event", win, min_p
        )

    df_model["target_sg_total"] = df_model["sg_total_tournament"]
    df_model["target_sg_per_round"] = df_model["sg_avg_round"]
    df_model["target_made_cut"] = df_model["made_cut"]
    df_model["target_top10"] = df_model["is_top10"]
    df_model["target_top20"] = df_model["is_top20"]
    df_model["target_win"] = df_model["is_win"]

    wind_round_frames = []
    for round_num, sg_col, wind_col in [
        (1, "sg_total_r1", "day0wind"),
        (2, "sg_total_r2", "day1wind"),
        (3, "sg_total_r3", "day2wind"),
    ]:
        frame = df_model[["name", "season", "tournament", "start", sg_col, wind_col]].copy()
        frame = frame.rename(columns={sg_col: "sg_round", wind_col: "wind"})
        frame["round_num"] = round_num
        wind_round_frames.append(frame)

    wind_rounds = pd.concat(wind_round_frames, ignore_index=True)
    wind_rounds["wind"] = pd.to_numeric(wind_rounds["wind"], errors="coerce")
    wind_rounds = wind_rounds.dropna(subset=["sg_round", "wind", "start"])
    wind_rounds = wind_rounds.sort_values(
        ["name", "start", "tournament", "round_num"]
    ).reset_index(drop=True)

    wind_rounds["is_high_wind"] = wind_rounds["wind"] > 15.0
    wind_rounds["sg_high_only"] = np.where(wind_rounds["is_high_wind"], wind_rounds["sg_round"], 0.0)
    wind_rounds["sg_normal_only"] = np.where(~wind_rounds["is_high_wind"], wind_rounds["sg_round"], 0.0)
    wind_rounds["cnt_high_only"] = wind_rounds["is_high_wind"].astype(int)
    wind_rounds["cnt_normal_only"] = (~wind_rounds["is_high_wind"]).astype(int)

    wind_group = wind_rounds.groupby("name", sort=False)
    wind_rounds["cum_sg_high_before"] = wind_group["sg_high_only"].cumsum() - wind_rounds["sg_high_only"]
    wind_rounds["cum_sg_normal_before"] = (
        wind_group["sg_normal_only"].cumsum() - wind_rounds["sg_normal_only"]
    )
    wind_rounds["high_wind_rounds_before"] = (
        wind_group["cnt_high_only"].cumsum() - wind_rounds["cnt_high_only"]
    )
    wind_rounds["normal_wind_rounds_before"] = (
        wind_group["cnt_normal_only"].cumsum() - wind_rounds["cnt_normal_only"]
    )

    wind_rounds["avg_sg_high_before"] = np.where(
        wind_rounds["high_wind_rounds_before"] > 0,
        wind_rounds["cum_sg_high_before"] / wind_rounds["high_wind_rounds_before"],
        np.nan,
    )
    wind_rounds["avg_sg_normal_before"] = np.where(
        wind_rounds["normal_wind_rounds_before"] > 0,
        wind_rounds["cum_sg_normal_before"] / wind_rounds["normal_wind_rounds_before"],
        np.nan,
    )
    wind_rounds["wind_premium_before"] = (
        wind_rounds["avg_sg_high_before"] - wind_rounds["avg_sg_normal_before"]
    )

    wind_event = (
        wind_rounds.sort_values(["name", "start", "tournament", "round_num"])
        .groupby(["name", "season", "tournament", "start"], as_index=False)
        .first()[
            [
                "name",
                "season",
                "tournament",
                "start",
                "wind_premium_before",
                "high_wind_rounds_before",
                "normal_wind_rounds_before",
            ]
        ]
    )

    df_model = df_model.drop(
        columns=[
            "wind_premium_before",
            "high_wind_rounds_before",
            "normal_wind_rounds_before",
        ],
        errors="ignore",
    )
    df_model = df_model.merge(
        wind_event,
        on=["name", "season", "tournament", "start"],
        how="left",
    )
    df_model["wind_feature_ready"] = (
        (df_model["high_wind_rounds_before"].fillna(0) >= 10)
        & (df_model["normal_wind_rounds_before"].fillna(0) >= 30)
    ).astype(int)

    df_model = df_model.drop(columns=["is_liv_event", "is_strong_field_event"], errors="ignore")

    return df_model


def assemble_feature_store(df_model: pd.DataFrame) -> pd.DataFrame:
    identity_cols = [
        "season",
        "start",
        "end",
        "tournament",
        "location",
        "name",
        "position_str",
        "position_num",
        "rounds_played",
        "Course Par",
    ]
    feature_cols = [
        "starts_before",
        "prev_avg_sg_total",
        "prev_avg_sg_round",
        "prev_avg_rounds_played",
        "prev_cut_rate",
        "prev_win_rate",
        "prev_top5_rate",
        "prev_top10_rate",
        "prev_top20_rate",
        "prev_avg_finish_num",
        "prev_avg_r1_sg",
        "prev_avg_r4_sg",
        "prev_avg_close_delta",
        "prev_sg_form_3",
        "prev_sg_form_5",
        "prev_sg_form_10",
        "prev_sg_form_20",
        "prev_form_trend_5v20",
        "prev_round_std_10",
        "event_starts_before",
        "prev_event_avg_sg_round",
        "prev_event_cut_rate",
        "prev_event_top20_rate",
        "field_strength_prev_avg_sg",
        "field_strength_median_prev_sg",
        "relative_skill_vs_field",
        "relative_skill_vs_field_median",
        "liv_share_last_10",
        "liv_share_last_20",
        "strong_field_share_last_10",
        "strong_field_share_last_20",
        "field_size",
        "wind_premium_before",
        "high_wind_rounds_before",
        "normal_wind_rounds_before",
        "wind_feature_ready",
    ]
    target_cols = [
        "target_sg_total",
        "target_sg_per_round",
        "target_made_cut",
        "target_top10",
        "target_top20",
        "target_win",
    ]
    selected = [c for c in identity_cols + feature_cols + target_cols if c in df_model.columns]
    feature_store = df_model[selected].copy()
    feature_store["history_5_plus"] = (feature_store["starts_before"] >= 5).astype(int)
    feature_store["history_20_plus"] = (feature_store["starts_before"] >= 20).astype(int)
    feature_store["dataset_split"] = np.select(
        [
            feature_store["start"] < pd.Timestamp("2023-07-01"),
            (feature_store["start"] >= pd.Timestamp("2023-07-01"))
            & (feature_store["start"] < pd.Timestamp("2025-01-01")),
            feature_store["start"] >= pd.Timestamp("2025-01-01"),
        ],
        ["train", "valid", "test"],
        default="train",
    )
    return feature_store


def build_and_save(
    out_path: str | Path | None = None,
    results_path: str | Path | None = None,
    weather_path: str | Path | None = None,
    weather_join: str = "left",
    supplement_path: Optional[Path] = None,
    use_supplement: bool = True,
) -> Path:
    out = Path(out_path or DEFAULT_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    rp = Path(results_path) if results_path else DEFAULT_RESULTS
    wp = Path(weather_path) if weather_path else DEFAULT_WEATHER
    df = load_base_frames(
        rp,
        wp,
        weather_join=weather_join,
        supplement_path=supplement_path,
        use_supplement=use_supplement,
    )
    df_model = build_model_frame(df)
    fs = assemble_feature_store(df_model)
    fs.to_csv(out, index=False)
    print(f"Wrote {out} shape={fs.shape} seasons {fs['season'].min()}–{fs['season'].max()}")
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--weather-join",
        choices=("left", "inner"),
        default="left",
        help="inner ≈ old notebook (~41k rows, 2009–2022 only); left = full TSV through 2025",
    )
    p.add_argument(
        "--no-supplement",
        action="store_true",
        help="Ignore pga_results_espn_supplement.tsv if present",
    )
    args = p.parse_args()
    build_and_save(
        weather_join=args.weather_join,
        use_supplement=not args.no_supplement,
    )
