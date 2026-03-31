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
    return df


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
    player_group = df_model.groupby("name", sort=False)

    df_model["starts_before"] = player_group.cumcount()
    df_model["prev_avg_sg_total"] = player_group["sg_total_tournament"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_avg_sg_round"] = player_group["sg_avg_round"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_avg_rounds_played"] = player_group["rounds_played"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_cut_rate"] = player_group["made_cut"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_win_rate"] = player_group["is_win"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_top5_rate"] = player_group["is_top5"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_top10_rate"] = player_group["is_top10"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_top20_rate"] = player_group["is_top20"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_avg_finish_num"] = player_group["position_num"].transform(
        lambda s: s.shift().expanding().mean()
    )

    df_model["prev_avg_r1_sg"] = player_group["sg_total_r1"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_avg_r4_sg"] = player_group["sg_total_r4"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_avg_close_delta"] = player_group["sg_close_delta"].transform(
        lambda s: s.shift().expanding().mean()
    )

    for window in [3, 5, 10, 20]:
        min_periods = 2 if window <= 5 else 5
        df_model[f"prev_sg_form_{window}"] = player_group["sg_avg_round"].transform(
            lambda s, w=window, mp=min_periods: s.shift().rolling(w, min_periods=mp).mean()
        )

    df_model["prev_form_trend_5v20"] = (
        df_model["prev_sg_form_5"] - df_model["prev_sg_form_20"]
    )
    df_model["prev_round_std_10"] = player_group["round_score_std"].transform(
        lambda s: s.shift().rolling(10, min_periods=5).mean()
    )

    event_group = df_model.groupby(["name", "tournament"], sort=False)
    df_model["event_starts_before"] = event_group.cumcount()
    df_model["prev_event_avg_sg_round"] = event_group["sg_avg_round"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_event_cut_rate"] = event_group["made_cut"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df_model["prev_event_top20_rate"] = event_group["is_top20"].transform(
        lambda s: s.shift().expanding().mean()
    )

    df_model["field_strength_prev_avg_sg"] = df_model.groupby(["season", "tournament"])[
        "prev_avg_sg_round"
    ].transform("mean")
    df_model["relative_skill_vs_field"] = (
        df_model["prev_avg_sg_round"] - df_model["field_strength_prev_avg_sg"]
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
        "relative_skill_vs_field",
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
            feature_store["start"] < pd.Timestamp("2022-01-01"),
            (feature_store["start"] >= pd.Timestamp("2022-01-01"))
            & (feature_store["start"] < pd.Timestamp("2024-01-01")),
            feature_store["start"] >= pd.Timestamp("2024-01-01"),
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
    build_and_save(weather_join=args.weather_join, use_supplement=not args.no_supplement)
