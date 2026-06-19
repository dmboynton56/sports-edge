#!/usr/bin/env python3
"""Update PGA tournament probabilities from completed rounds and live scores."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pga.live_leaderboard import (  # noqa: E402
    fetch_live_leaderboard,
    format_cut_line,
    normalize_name,
    rounds_completed_from_leaderboard,
)


DEFAULT_KEY = "us_open_2026"


def default_pred_csv(tournament_key: str) -> Path:
    return ROOT / "notebooks" / "cache" / f"{tournament_key}_predictions.csv"


def default_out_csv(tournament_key: str) -> Path:
    return ROOT / "notebooks" / "cache" / f"{tournament_key}_midtournament.csv"


def load_pretournament_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pre-tournament predictions not found: {path}")
    return pd.read_csv(path)


def build_name_map(pred_names: list[str], espn_names: list[str]) -> dict[str, str]:
    pred_norm = {normalize_name(name): name for name in pred_names}
    mapping: dict[str, str] = {}
    for espn_name in espn_names:
        norm = normalize_name(espn_name)
        if norm in pred_norm:
            mapping[espn_name] = pred_norm[norm]
            continue
        parts = norm.split()
        if len(parts) < 2:
            continue
        for pred_norm_name, original in pred_norm.items():
            pred_parts = pred_norm_name.split()
            if len(pred_parts) >= 2 and pred_parts[-1] == parts[-1] and pred_parts[0][0] == parts[0][0]:
                mapping[espn_name] = original
                break
    return mapping


def run_mc_from_actual(
    actual_totals: np.ndarray,
    updated_sg_per_round: np.ndarray,
    n_remaining_rounds: int,
    n_sims: int,
    player_stds: np.ndarray,
    *,
    course_par: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_players = len(actual_totals)
    sim_finals = np.tile(actual_totals, (n_sims, 1)).astype(np.float64)

    for _ in range(max(n_remaining_rounds, 0)):
        round_scores = rng.normal(
            loc=course_par - updated_sg_per_round,
            scale=player_stds,
            size=(n_sims, n_players),
        )
        sim_finals += round_scores

    win = np.zeros(n_players)
    t5 = np.zeros(n_players)
    t10 = np.zeros(n_players)
    t20 = np.zeros(n_players)
    for sim in range(n_sims):
        ranks = np.argsort(sim_finals[sim])
        win[ranks[0]] += 1
        t5[ranks[: min(5, n_players)]] += 1
        t10[ranks[: min(10, n_players)]] += 1
        t20[ranks[: min(20, n_players)]] += 1

    denom = float(n_sims)
    return win / denom, t5 / denom, t10 / denom, t20 / denom


def run_make_cut_from_actual(
    actual_totals: np.ndarray,
    updated_sg_per_round: np.ndarray,
    n_rounds_to_cut: int,
    n_sims: int,
    player_stds: np.ndarray,
    *,
    course_par: int,
    cut_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate make-cut probability from the current completed-round state."""

    n_players = len(actual_totals)
    if n_rounds_to_cut <= 0:
        return np.ones(n_players)

    sim_cut_totals = np.tile(actual_totals, (n_sims, 1)).astype(np.float64)
    for _ in range(n_rounds_to_cut):
        round_scores = rng.normal(
            loc=course_par - updated_sg_per_round,
            scale=player_stds,
            size=(n_sims, n_players),
        )
        sim_cut_totals += round_scores

    made_cut = np.zeros(n_players)
    for sim in range(n_sims):
        totals = sim_cut_totals[sim]
        if n_players <= cut_size:
            made_cut += 1
            continue
        cut_line = np.partition(totals, cut_size - 1)[cut_size - 1]
        made_cut += totals <= cut_line
    return made_cut / float(n_sims)


def _safe_prob(row: pd.Series, *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value):
            return value
    return None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _round_score(rounds: dict[Any, Any], round_no: int) -> float | None:
    value = rounds.get(round_no)
    if value is None:
        value = rounds.get(str(round_no))
    score = _safe_float(value)
    if score is None or score <= 0:
        return None
    return score


def _completed_round_scores(player: dict[str, Any], rounds_completed: int) -> list[float]:
    rounds = player.get("rounds") or {}
    holes_by_round = player.get("roundHoles") or {}
    return [
        score
        for round_no in range(1, rounds_completed + 1)
        if holes_by_round.get(round_no, 18) >= 18 and (score := _round_score(rounds, round_no)) is not None
    ]


def _format_to_par_value(value: float) -> str:
    if value == 0:
        return "E"
    if float(value).is_integer():
        number = int(value)
        return f"+{number}" if number > 0 else str(number)
    return f"+{value:.1f}" if value > 0 else f"{value:.1f}"


def _total_strokes(player: dict[str, Any], *, course_par: int, rounds_completed: int) -> float:
    completed_scores = _completed_round_scores(player, rounds_completed)
    if len(completed_scores) == rounds_completed:
        return float(sum(completed_scores))
    if completed_scores:
        missing_rounds = rounds_completed - len(completed_scores)
        return float(sum(completed_scores) + (course_par * missing_rounds))
    return float(course_par * max(rounds_completed, 1))


def _completed_to_par(player: dict[str, Any], *, course_par: int, rounds_completed: int) -> float:
    total = _total_strokes(player, course_par=course_par, rounds_completed=rounds_completed)
    return total - (course_par * max(rounds_completed, 1))


def _active_players_for_completed_state(
    players: list[dict[str, Any]],
    *,
    rounds_completed: int,
    cut_after_round: int,
    cut_size: int,
    course_par: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None, bool]:
    active: list[dict[str, Any]] = []
    inactive: list[dict[str, Any]] = []
    inactive_statuses = {"WD", "DQ", "DNS", "CUT", "MDF", ""}

    for player in players:
        if str(player.get("toPar") or "").upper() in inactive_statuses:
            inactive.append(player)
            continue
        completed = dict(player)
        completed["_completed_total_strokes"] = _total_strokes(
            player,
            course_par=course_par,
            rounds_completed=rounds_completed,
        )
        completed["_completed_to_par"] = _completed_to_par(
            player,
            course_par=course_par,
            rounds_completed=rounds_completed,
        )
        completed["_completed_to_par_display"] = _format_to_par_value(float(completed["_completed_to_par"]))
        active.append(completed)

    active.sort(key=lambda player: (player["_completed_to_par"], player["_completed_total_strokes"], player["player"]))
    if rounds_completed < cut_after_round:
        return active, inactive, None, False
    if not active:
        return [], inactive, None, True
    if len(active) <= cut_size:
        return active, inactive, float(active[-1]["_completed_to_par"]), True

    cut_line = float(active[cut_size - 1]["_completed_to_par"])
    made = [player for player in active if float(player["_completed_to_par"]) <= cut_line]
    missed = [player for player in active if float(player["_completed_to_par"]) > cut_line]
    missed.extend(inactive)
    return made, missed, cut_line, True


def _assign_tied_positions(df: pd.DataFrame) -> pd.Series:
    pos_display = [""] * len(df)
    index = 0
    while index < len(df):
        end = index
        while end + 1 < len(df) and df.iloc[end + 1]["to_par"] == df.iloc[index]["to_par"]:
            end += 1
        rank = index + 1
        tied = end > index
        for pos in range(index, end + 1):
            pos_display[pos] = f"T{rank}" if tied else str(rank)
        index = end + 1
    return pd.Series(pos_display, index=df.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update PGA tournament predictions with completed-round state.")
    parser.add_argument("--tournament-key", default=DEFAULT_KEY)
    parser.add_argument("--event-name", default="PGA Tournament")
    parser.add_argument("--course-par", type=int, default=70)
    parser.add_argument("--cut-size", type=int, default=65)
    parser.add_argument("--cut-after-round", type=int, default=2)
    parser.add_argument("--total-rounds", type=int, default=4)
    parser.add_argument("--pred-csv", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--espn-match", action="append", default=[])
    parser.add_argument("--actual-weight", type=float, default=0.40)
    parser.add_argument("--n-sims", type=int, default=50000)
    parser.add_argument("--sg-std-default", type=float, default=2.5)
    parser.add_argument("--remaining-rounds", type=int)
    parser.add_argument("--seed", type=int, default=20260618)
    return parser.parse_args()


def main() -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    args = parse_args()
    pred_csv = args.pred_csv or default_pred_csv(args.tournament_key)
    out_csv = args.out_csv or default_out_csv(args.tournament_key)

    print("Fetching live ESPN leaderboard...")
    leaderboard = fetch_live_leaderboard(espn_match=args.espn_match)
    if not leaderboard:
        raise SystemExit(f"Could not fetch ESPN leaderboard for {args.tournament_key}.")

    rounds_completed = rounds_completed_from_leaderboard(leaderboard, total_rounds=args.total_rounds)
    if rounds_completed <= 0:
        raise SystemExit("No completed rounds available for mid-tournament update.")
    remaining_rounds = args.remaining_rounds
    if remaining_rounds is None:
        remaining_rounds = max(args.total_rounds - rounds_completed, 0)

    active, out_players, cut_line, cut_applied = _active_players_for_completed_state(
        leaderboard["players"],
        rounds_completed=rounds_completed,
        cut_after_round=args.cut_after_round,
        cut_size=args.cut_size,
        course_par=args.course_par,
    )

    print(f"Event: {leaderboard['event']}")
    print(f"Status: {leaderboard['status']} (Round {leaderboard['currentRound']})")
    print(f"Rounds completed: {rounds_completed}, Remaining: {remaining_rounds}")
    if cut_applied:
        print(f"Cut line: {format_cut_line(cut_line)} | Made cut: {len(active)} | Out: {len(out_players)}")
    else:
        print(f"Cut not applied until after Round {args.cut_after_round}; active players: {len(active)}")

    print("\nLoading pre-tournament predictions...")
    pre_df = load_pretournament_predictions(pred_csv)
    pred_names = pre_df["player"].tolist()
    espn_names = [player["player"] for player in active]
    name_map = build_name_map(pred_names, espn_names)
    print(f"Name matches: {len(name_map)}/{len(espn_names)} ESPN players matched to predictions")

    unmatched = [name for name in espn_names if name not in name_map]
    if unmatched:
        print(f"Unmatched ESPN players: {unmatched}")

    pre_by_name = pre_df.set_index("player")
    rows: list[dict[str, Any]] = []
    for player in active:
        espn_name = player["player"]
        pred_name = name_map.get(espn_name)
        pre_sg = 0.0
        pre_win = None
        pre_top5 = None
        pre_top10 = None
        pre_top20 = None
        if pred_name and pred_name in pre_by_name.index:
            pre_row = pre_by_name.loc[pred_name]
            pre_sg = float(pre_row.get("exp_sg_per_round", 0.0))
            pre_win = _safe_float(_safe_prob(pre_row, "best_calibrated_target_win_prob", "sim_win_pct"))
            if pre_win is not None and pre_win > 1:
                pre_win = pre_win / 100.0
            pre_top5 = _safe_prob(pre_row, "sim_top5_pct")
            pre_top10 = _safe_prob(pre_row, "best_calibrated_target_top10_prob")
            pre_top20 = _safe_prob(pre_row, "best_calibrated_target_top20_prob")

        total_strokes = _total_strokes(player, course_par=args.course_par, rounds_completed=rounds_completed)
        to_par_val = total_strokes - (args.course_par * rounds_completed)
        actual_sg_per_round = (args.course_par * rounds_completed - total_strokes) / max(rounds_completed, 1)
        rounds = player.get("rounds") or {}
        row = {
            "player": espn_name,
            "pred_name": pred_name or espn_name,
            "total_strokes": total_strokes,
            "to_par": to_par_val,
            "to_par_display": _format_to_par_value(to_par_val),
            "actual_sg_per_round": actual_sg_per_round,
            "pre_sg_per_round": pre_sg,
            "pre_win_prob": pre_win if pre_win is not None and not pd.isna(pre_win) else None,
            "pre_top5_pct": pre_top5 if pre_top5 is not None and not pd.isna(pre_top5) else None,
            "pre_top10_prob": pre_top10 if pre_top10 is not None and not pd.isna(pre_top10) else None,
            "pre_top20_prob": pre_top20 if pre_top20 is not None and not pd.isna(pre_top20) else None,
        }
        for round_no in range(1, args.total_rounds + 1):
            row[f"r{round_no}"] = _round_score(rounds, round_no) if round_no <= rounds_completed else None
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No active players available for simulation.")

    alpha = args.actual_weight
    df["updated_sg_per_round"] = alpha * df["actual_sg_per_round"] + (1 - alpha) * df["pre_sg_per_round"]
    player_stds = np.full(len(df), args.sg_std_default)
    rng = np.random.default_rng(args.seed + rounds_completed)
    win_p, t5_p, t10_p, t20_p = run_mc_from_actual(
        df["total_strokes"].to_numpy(dtype=float),
        df["updated_sg_per_round"].to_numpy(dtype=float),
        remaining_rounds,
        args.n_sims,
        player_stds,
        course_par=args.course_par,
        rng=rng,
    )
    make_cut_p = run_make_cut_from_actual(
        df["total_strokes"].to_numpy(dtype=float),
        df["updated_sg_per_round"].to_numpy(dtype=float),
        max(args.cut_after_round - rounds_completed, 0),
        args.n_sims,
        player_stds,
        course_par=args.course_par,
        cut_size=args.cut_size,
        rng=np.random.default_rng(args.seed + 1000 + rounds_completed),
    )

    df["sim_make_cut_pct"] = 100 * make_cut_p
    df["sim_win_pct"] = 100 * win_p
    df["sim_top5_pct"] = 100 * t5_p
    df["sim_top10_pct"] = 100 * t10_p
    df["sim_top20_pct"] = 100 * t20_p
    df = df.sort_values("to_par").reset_index(drop=True)
    df["current_pos"] = range(1, len(df) + 1)
    df["current_pos_display"] = _assign_tied_positions(df)

    pre_rank_col = "best_calibrated_target_win_prob" if "best_calibrated_target_win_prob" in pre_df.columns else "sim_win_pct"
    pre_df_sorted = pre_df.sort_values(pre_rank_col, ascending=False).reset_index(drop=True)
    pre_rank_map = {row["player"]: index + 1 for index, row in pre_df_sorted.iterrows()}
    df["pre_rank"] = df["pred_name"].map(pre_rank_map)
    df["rank_change"] = df["pre_rank"] - df["current_pos"]

    print("\n" + "=" * 120)
    print(f"{args.event_name.upper()} - MID-TOURNAMENT UPDATE (after Round {rounds_completed})")
    print(f"Actual weight: {alpha:.0%} | {len(df)} simulated players | {args.n_sims:,} sims x {remaining_rounds} rounds remaining")
    print("=" * 120)
    for _, row in df.head(20).iterrows():
        print(
            f"{row['current_pos_display']:<5} {row['player']:<28} {row['to_par_display']:>6} "
            f"UpdSG/R {row['updated_sg_per_round']:+.2f} Win {row['sim_win_pct']:.1f}% "
            f"T10 {row['sim_top10_pct']:.1f}% T20 {row['sim_top20_pct']:.1f}%"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [
        "current_pos",
        "current_pos_display",
        "player",
        "pred_name",
        "to_par",
        "to_par_display",
        *[f"r{round_no}" for round_no in range(1, args.total_rounds + 1)],
        "total_strokes",
        "actual_sg_per_round",
        "pre_sg_per_round",
        "updated_sg_per_round",
        "sim_win_pct",
        "sim_make_cut_pct",
        "sim_top5_pct",
        "sim_top10_pct",
        "sim_top20_pct",
        "pre_win_prob",
        "pre_top5_pct",
        "pre_top10_prob",
        "pre_top20_prob",
        "pre_rank",
        "rank_change",
    ]
    df[[col for col in export_cols if col in df.columns]].to_csv(out_csv, index=False)

    state_key = f"{args.tournament_key}:R{rounds_completed}"
    meta = {
        "type": "midtournament",
        "tournament_key": args.tournament_key,
        "event": leaderboard["event"],
        "status": leaderboard["status"],
        "rounds_completed": rounds_completed,
        "remaining_rounds": remaining_rounds,
        "round_state_key": state_key,
        "cut_applied": cut_applied,
        "cut_after_round": args.cut_after_round,
        "cut_size": args.cut_size,
        "cut_line": format_cut_line(cut_line),
        "made_cut": len(active) if cut_applied else None,
        "active_players": len(active),
        "out_players": len(out_players),
        "n_sims": args.n_sims,
        "actual_weight": alpha,
        "pretournament_weight": 1 - alpha,
        "sg_std": args.sg_std_default,
        "seed": args.seed,
        "round_complete_validated": True,
        "score_source": "espn_completed_round_scores_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pretournament_predictions": str(pred_csv),
        "leaderboard_fetched_at": leaderboard.get("fetchedAt", ""),
    }
    meta_path = out_csv.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nWrote {out_csv}")
    print(f"Wrote {meta_path}")
    return df, meta, active, out_players, leaderboard


if __name__ == "__main__":
    main()
