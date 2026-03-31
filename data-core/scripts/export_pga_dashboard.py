#!/usr/bin/env python3
"""
Bundle Masters predictions + ESPN 2026 supplement into web/public/data/pga_masters_dashboard.json.

  cd data-core && .venv/bin/python scripts/export_pga_dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = Path(project_root).parent

ARCHIVE = Path(project_root) / "src" / "data" / "archive"
SUPP = ARCHIVE / "pga_results_espn_supplement.tsv"
PRED = Path(project_root) / "notebooks" / "cache" / "masters_2026_predictions.csv"
PRED_META = PRED.with_suffix(".meta.json")
OUT = repo_root / "web" / "public" / "data" / "pga_masters_dashboard.json"


def load_predictions() -> tuple[list[dict], dict]:
    meta = {}
    if PRED_META.exists():
        meta = json.loads(PRED_META.read_text())
    if not PRED.exists():
        return [], meta
    df = pd.read_csv(PRED)
    return df.to_dict(orient="records"), meta


def supplement_summaries(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """Per-tournament rollup + last starts per player."""
    if df.empty:
        return [], {}
    df = df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    events = []
    for (tourn, start), g in df.groupby(["tournament", "start"], sort=False):
        events.append(
            {
                "tournament": str(tourn),
                "start": str(start.date()) if pd.notna(start) else "",
                "players": int(len(g)),
            }
        )
    events.sort(key=lambda x: x["start"])

    by_player: dict[str, list] = defaultdict(list)
    for _, r in df.sort_values("start", ascending=False).iterrows():
        by_player[str(r["name"])].append(
            {
                "tournament": str(r["tournament"]),
                "start": str(r["start"])[:10] if pd.notna(r["start"]) else "",
                "position": str(r["position"]),
                "scoreToPar": str(r["score"]),
                "r1": r["round1"] if pd.notna(r.get("round1")) else None,
                "r2": r["round2"] if pd.notna(r.get("round2")) else None,
                "r3": r["round3"] if pd.notna(r.get("round3")) else None,
                "r4": r["round4"] if pd.notna(r.get("round4")) else None,
                "total": int(r["total"]) if pd.notna(r.get("total")) and str(r["total"]) else None,
            }
        )

    recent_by_player = {k: v[:6] for k, v in by_player.items()}
    return events, recent_by_player


def _rel_to_repo(p: str) -> str:
    try:
        return str(Path(p).relative_to(repo_root))
    except ValueError:
        return p


def main() -> None:
    preds, pred_meta = load_predictions()
    if isinstance(pred_meta.get("feature_store"), str):
        pred_meta = {**pred_meta, "feature_store": _rel_to_repo(pred_meta["feature_store"])}
    supplement_meta = {"path": _rel_to_repo(str(SUPP)), "rows": 0}
    events: list = []
    recent_by_player: dict = {}

    if SUPP.exists():
        sdf = pd.read_csv(SUPP, sep="\t")
        supplement_meta["rows"] = len(sdf)
        supplement_meta["seasons"] = sorted(sdf["season"].dropna().unique().tolist()) if "season" in sdf.columns else []
        events, recent_by_player = supplement_summaries(sdf)

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "predictions": preds,
        "predictionMeta": pred_meta,
        "espnSupplement": supplement_meta,
        "tournaments2026": events,
        "recentByPlayer": recent_by_player,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT} ({len(preds)} predictions, {len(events)} events, {len(recent_by_player)} players w/ form)")


if __name__ == "__main__":
    main()
