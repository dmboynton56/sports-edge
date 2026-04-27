#!/usr/bin/env python3
"""
Mid-tournament Masters predictions — incorporates actual round scores and cut.

Fetches the live ESPN leaderboard, filters to players who made the cut,
Bayesian-updates expected SG/R by blending pre-tournament model with actual
tournament performance, then re-runs Monte Carlo for the remaining rounds.

  cd data-core
  .venv/bin/python scripts/update_masters_midtournament.py
  .venv/bin/python scripts/update_masters_midtournament.py --actual-weight 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.export_pga_dashboard import fetch_live_leaderboard, _sort_key_to_par

PAR = 72
PRED_CSV = Path(project_root) / "notebooks" / "cache" / "masters_2026_predictions.csv"
PRED_META = PRED_CSV.with_suffix(".meta.json")
OUT_CSV = Path(project_root) / "notebooks" / "cache" / "masters_2026_midtournament.csv"
OUT_META = OUT_CSV.with_suffix(".meta.json")


def load_pretournament_predictions() -> pd.DataFrame:
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Pre-tournament predictions not found: {PRED_CSV}")
    return pd.read_csv(PRED_CSV)


def normalize_name(name: str) -> str:
    """Normalize player names for fuzzy matching between ESPN and our predictions."""
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.strip().lower()


def build_name_map(pred_names: List[str], espn_names: List[str]) -> Dict[str, str]:
    """Map ESPN display names -> prediction player names via normalized matching."""
    pred_norm = {normalize_name(n): n for n in pred_names}
    mapping: Dict[str, str] = {}
    for espn_name in espn_names:
        norm = normalize_name(espn_name)
        if norm in pred_norm:
            mapping[espn_name] = pred_norm[norm]
            continue
        # Try last-name + first-initial match
        parts = norm.split()
        if len(parts) >= 2:
            for pn, orig in pred_norm.items():
                pp = pn.split()
                if len(pp) >= 2 and pp[-1] == parts[-1] and pp[0][0] == parts[0][0]:
                    mapping[espn_name] = orig
                    break
    return mapping


def determine_cut(
    players: List[Dict[str, Any]], top_n: int = 50
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """
    Determine the cut line (top N and ties at the Masters).
    Returns (made_cut, missed_cut, cut_score_to_par).
    """
    valid = [p for p in players if p.get("toPar") not in ("WD", "DQ", "DNS", "")]
    valid.sort(key=lambda p: (_sort_key_to_par(p["toPar"]), p.get("totalStrokes") or 999))

    if len(valid) <= top_n:
        return valid, [], _sort_key_to_par(valid[-1]["toPar"]) if valid else 0.0

    cutoff_score = _sort_key_to_par(valid[top_n - 1]["toPar"])
    made = [p for p in valid if _sort_key_to_par(p["toPar"]) <= cutoff_score]
    missed = [p for p in valid if _sort_key_to_par(p["toPar"]) > cutoff_score]

    wd_dq = [p for p in players if p.get("toPar") in ("WD", "DQ", "DNS")]
    missed.extend(wd_dq)

    return made, missed, cutoff_score


def run_mc_from_actual(
    actual_totals: np.ndarray,
    updated_sg_per_round: np.ndarray,
    n_remaining_rounds: int,
    n_sims: int,
    player_stds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation starting from actual stroke totals.
    Simulates remaining rounds and ranks by final total (lowest wins).
    """
    n_players = len(actual_totals)
    sim_finals = np.tile(actual_totals, (n_sims, 1)).astype(np.float64)

    for _ in range(n_remaining_rounds):
        round_scores = np.random.normal(
            loc=PAR - updated_sg_per_round,
            scale=player_stds,
            size=(n_sims, n_players),
        )
        sim_finals += round_scores

    win = np.zeros(n_players)
    t5 = np.zeros(n_players)
    t10 = np.zeros(n_players)
    t20 = np.zeros(n_players)

    for s in range(n_sims):
        ranks = np.argsort(sim_finals[s])
        win[ranks[0]] += 1
        for j in range(min(5, n_players)):
            t5[ranks[j]] += 1
        for j in range(min(10, n_players)):
            t10[ranks[j]] += 1
        for j in range(min(20, n_players)):
            t20[ranks[j]] += 1

    return win / n_sims, t5 / n_sims, t10 / n_sims, t20 / n_sims


def main() -> None:
    ap = argparse.ArgumentParser(description="Mid-tournament Masters prediction update")
    ap.add_argument(
        "--actual-weight",
        type=float,
        default=0.40,
        help="Weight for actual tournament SG vs pre-tournament model (0-1). Higher = trust actuals more.",
    )
    ap.add_argument("--n-sims", type=int, default=50000)
    ap.add_argument("--sg-std-default", type=float, default=2.5)
    ap.add_argument(
        "--remaining-rounds",
        type=int,
        default=None,
        help="Override remaining rounds (auto-detected from ESPN round)",
    )
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    print("Fetching live ESPN leaderboard...")
    leaderboard = fetch_live_leaderboard()
    if not leaderboard:
        print("ERROR: Could not fetch ESPN leaderboard. Is the Masters in progress?")
        sys.exit(1)

    current_round = leaderboard["currentRound"]
    status = leaderboard["status"]
    event = leaderboard["event"]
    rounds_completed = current_round if "Complete" in status else current_round - 1
    remaining_rounds = args.remaining_rounds or (4 - rounds_completed)

    print(f"Event: {event}")
    print(f"Status: {status} (Round {current_round})")
    print(f"Rounds completed: {rounds_completed}, Remaining: {remaining_rounds}")
    print(f"Players on leaderboard: {len(leaderboard['players'])}")

    espn_players = leaderboard["players"]
    made_cut, missed_cut, cut_line = determine_cut(espn_players, top_n=50)
    print(f"\nCut line: {'+' if cut_line > 0 else ''}{int(cut_line) if cut_line == int(cut_line) else cut_line}")
    print(f"Made cut: {len(made_cut)} players")
    print(f"Missed cut: {len(missed_cut)} players")

    print("\nLoading pre-tournament predictions...")
    pre_df = load_pretournament_predictions()
    pred_names = pre_df["player"].tolist()
    espn_names = [p["player"] for p in made_cut]
    name_map = build_name_map(pred_names, espn_names)

    print(f"Name matches: {len(name_map)}/{len(espn_names)} ESPN players matched to predictions")
    unmatched = [n for n in espn_names if n not in name_map]
    if unmatched:
        print(f"Unmatched ESPN players: {unmatched}")

    pre_by_name = pre_df.set_index("player")

    rows = []
    for p in made_cut:
        espn_name = p["player"]
        pred_name = name_map.get(espn_name)
        pre_sg = 0.0
        pre_win = None
        pre_top5 = None
        pre_top10 = None
        pre_top20 = None

        if pred_name and pred_name in pre_by_name.index:
            pre_row = pre_by_name.loc[pred_name]
            pre_sg = float(pre_row.get("exp_sg_per_round", 0.0))
            pre_win = pre_row.get("best_calibrated_target_win_prob")
            pre_top5 = pre_row.get("sim_top5_pct")
            pre_top10 = pre_row.get("best_calibrated_target_top10_prob")
            pre_top20 = pre_row.get("best_calibrated_target_top20_prob")
            if pd.isna(pre_win):
                pre_win = pre_row.get("sim_win_pct")
                if pre_win is not None and not pd.isna(pre_win):
                    pre_win = pre_win / 100.0

        total_strokes = p.get("totalStrokes") or 0
        to_par_val = _sort_key_to_par(p["toPar"])
        actual_sg_per_round = (PAR * rounds_completed - total_strokes) / max(rounds_completed, 1)

        rows.append({
            "player": espn_name,
            "pred_name": pred_name or espn_name,
            "r1": p["rounds"].get(1),
            "r2": p["rounds"].get(2),
            "r3": p["rounds"].get(3),
            "r4": p["rounds"].get(4),
            "total_strokes": total_strokes,
            "to_par": to_par_val,
            "to_par_display": p["toPar"],
            "actual_sg_per_round": actual_sg_per_round,
            "pre_sg_per_round": pre_sg,
            "pre_win_prob": pre_win if pre_win is not None and not pd.isna(pre_win) else None,
            "pre_top5_pct": pre_top5 if pre_top5 is not None and not pd.isna(pre_top5) else None,
            "pre_top10_prob": pre_top10 if pre_top10 is not None and not pd.isna(pre_top10) else None,
            "pre_top20_prob": pre_top20 if pre_top20 is not None and not pd.isna(pre_top20) else None,
        })

    df = pd.DataFrame(rows)

    alpha = args.actual_weight
    df["updated_sg_per_round"] = alpha * df["actual_sg_per_round"] + (1 - alpha) * df["pre_sg_per_round"]

    actual_totals = df["total_strokes"].values.astype(np.float64)
    updated_sg = df["updated_sg_per_round"].values.astype(np.float64)

    player_stds = np.full(len(df), args.sg_std_default)

    print(f"\nRunning Monte Carlo: {args.n_sims:,} sims × {remaining_rounds} remaining rounds")
    print(f"Bayesian blend: {alpha:.0%} actual + {1-alpha:.0%} pre-tournament model")

    win_p, t5_p, t10_p, t20_p = run_mc_from_actual(
        actual_totals, updated_sg, remaining_rounds, args.n_sims, player_stds,
    )

    df["sim_win_pct"] = 100 * win_p
    df["sim_top5_pct"] = 100 * t5_p
    df["sim_top10_pct"] = 100 * t10_p
    df["sim_top20_pct"] = 100 * t20_p

    df = df.sort_values("to_par").reset_index(drop=True)
    df["current_pos"] = range(1, len(df) + 1)

    # Assign tied positions
    i = 0
    pos_display = [""] * len(df)
    while i < len(df):
        j = i
        while j + 1 < len(df) and df.iloc[j + 1]["to_par"] == df.iloc[i]["to_par"]:
            j += 1
        rank = i + 1
        tied = j > i
        for k in range(i, j + 1):
            pos_display[k] = f"T{rank}" if tied else str(rank)
        i = j + 1
    df["current_pos_display"] = pos_display

    # Movement vs pre-tournament rank (by win prob)
    pre_df_sorted = pre_df.sort_values(
        "best_calibrated_target_win_prob" if "best_calibrated_target_win_prob" in pre_df.columns else "sim_win_pct",
        ascending=False,
    ).reset_index(drop=True)
    pre_rank_map = {row["player"]: idx + 1 for idx, row in pre_df_sorted.iterrows()}

    df["pre_rank"] = df["pred_name"].map(pre_rank_map)
    df["rank_change"] = df["pre_rank"] - df["current_pos"]

    print("\n" + "=" * 130)
    print(f"2026 MASTERS — MID-TOURNAMENT UPDATE (after Round {rounds_completed})")
    print(f"Actual weight: {alpha:.0%} | {len(df)} players made cut | {args.n_sims:,} sims × {remaining_rounds} rounds remaining")
    print("=" * 130)

    hdr = (
        f"{'Pos':<5} {'Player':<28} {'To Par':>6} {'R1':>4} {'R2':>4}"
        f" {'ActSG/R':>8} {'PreSG/R':>8} {'UpdSG/R':>8}"
        f" {'Win%':>7} {'T5%':>7} {'T10%':>7} {'T20%':>7}"
        f" {'PreRk':>6} {'Chg':>5}"
    )
    print(hdr)
    print("-" * len(hdr))

    for _, r in df.iterrows():
        pre_rk = f"{int(r['pre_rank'])}" if pd.notna(r.get("pre_rank")) else "—"
        chg = ""
        if pd.notna(r.get("rank_change")):
            rc = int(r["rank_change"])
            chg = f"+{rc}" if rc > 0 else str(rc) if rc < 0 else "="

        print(
            f"{r['current_pos_display']:<5} {r['player']:<28} {r['to_par_display']:>6}"
            f" {r['r1'] or '-':>4} {r['r2'] or '-':>4}"
            f" {r['actual_sg_per_round']:>+8.2f} {r['pre_sg_per_round']:>+8.2f} {r['updated_sg_per_round']:>+8.2f}"
            f" {r['sim_win_pct']:>6.1f}% {r['sim_top5_pct']:>6.1f}% {r['sim_top10_pct']:>6.1f}% {r['sim_top20_pct']:>6.1f}%"
            f" {pre_rk:>6} {chg:>5}"
        )

    # Biggest movers
    movers = df.dropna(subset=["rank_change"]).copy()
    if not movers.empty:
        print("\n" + "=" * 80)
        print("BIGGEST MOVERS (pre-tournament rank → current position)")
        print("=" * 80)
        risers = movers.nlargest(5, "rank_change")
        fallers = movers.nsmallest(5, "rank_change")
        print("\nRisers:")
        for _, r in risers.iterrows():
            if r["rank_change"] > 0:
                print(f"  {r['player']:<28} {int(r['pre_rank']):>3} → {r['current_pos_display']:<4} (+{int(r['rank_change'])})")
        print("\nFallers:")
        for _, r in fallers.iterrows():
            if r["rank_change"] < 0:
                print(f"  {r['player']:<28} {int(r['pre_rank']):>3} → {r['current_pos_display']:<4} ({int(r['rank_change'])})")

    # Notable cuts
    print("\n" + "=" * 80)
    print(f"NOTABLE PLAYERS WHO MISSED THE CUT ({len(missed_cut)} total)")
    print("=" * 80)
    notable_cuts = []
    for p in missed_cut:
        espn_name = p["player"]
        pred_name = name_map.get(espn_name) if espn_name in name_map else None
        pre_rank = pre_rank_map.get(pred_name) if pred_name else None
        if pre_rank and pre_rank <= 30:
            notable_cuts.append((espn_name, pre_rank, p["toPar"], p.get("totalStrokes", 0)))
    notable_cuts.sort(key=lambda x: x[1])
    # Also try matching missed_cut players
    missed_name_map = build_name_map(pred_names, [p["player"] for p in missed_cut])
    for p in missed_cut:
        espn_name = p["player"]
        pred_name = missed_name_map.get(espn_name)
        pre_rank = pre_rank_map.get(pred_name) if pred_name else None
        if pre_rank and pre_rank <= 30:
            if not any(n[0] == espn_name for n in notable_cuts):
                notable_cuts.append((espn_name, pre_rank, p["toPar"], p.get("totalStrokes", 0)))
    notable_cuts.sort(key=lambda x: x[1])
    for name, rank, to_par, total in notable_cuts:
        print(f"  {name:<28} Pre-tournament rank: {rank:>3}  Score: {to_par} ({total})")
    if not notable_cuts:
        for p in missed_cut[:10]:
            print(f"  {p['player']:<28} Score: {p['toPar']} ({p.get('totalStrokes', '')})")

    out_path = args.out_csv or OUT_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "current_pos", "current_pos_display", "player", "pred_name",
        "to_par", "to_par_display", "r1", "r2", "total_strokes",
        "actual_sg_per_round", "pre_sg_per_round", "updated_sg_per_round",
        "sim_win_pct", "sim_top5_pct", "sim_top10_pct", "sim_top20_pct",
        "pre_win_prob", "pre_top5_pct", "pre_top10_prob", "pre_top20_prob",
        "pre_rank", "rank_change",
    ]
    existing = [c for c in export_cols if c in df.columns]
    df[existing].to_csv(out_path, index=False)

    meta = {
        "type": "midtournament",
        "event": event,
        "status": status,
        "rounds_completed": rounds_completed,
        "remaining_rounds": remaining_rounds,
        "cut_line": f"+{int(cut_line)}" if cut_line > 0 else ("E" if cut_line == 0 else str(int(cut_line))),
        "made_cut": len(made_cut),
        "missed_cut": len(missed_cut),
        "n_sims": args.n_sims,
        "actual_weight": alpha,
        "pretournament_weight": 1 - alpha,
        "sg_std": args.sg_std_default,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pretournament_predictions": str(PRED_CSV),
        "leaderboard_fetched_at": leaderboard.get("fetchedAt", ""),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nWrote {out_path}")
    print(f"Wrote {meta_path}")

    return df, meta, made_cut, missed_cut, leaderboard


if __name__ == "__main__":
    main()
