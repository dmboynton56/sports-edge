#!/usr/bin/env python3
"""
Audit Masters field coverage: merged results TSVs vs feature store vs inference fallbacks.

  cd data-core && .venv/bin/python scripts/audit_masters_field_data.py
  .venv/bin/python scripts/audit_masters_field_data.py --field-file path.txt --as-of 2026-04-09
  .venv/bin/python scripts/audit_masters_field_data.py --json-out notebooks/cache/masters_field_audit.json

Checks per player:
  - Row counts and last start in main + supplement TSVs (merged, deduped like build step)
  - Last N events before --as-of
  - Feature store: rows for name, max(start) before as_of (matches live inference row)
  - WARN if predict would use Masters-only fallback (name missing from store)
  - WARN if any store row has start >= as_of (should not be used after as_of fix)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import predict_masters_tournament as pmt  # noqa: E402

DEFAULT_RESULTS_TSV = pmt.DEFAULT_RESULTS_TSV
DEFAULT_RESULTS_SUPPLEMENT = pmt.DEFAULT_RESULTS_SUPPLEMENT
DEFAULT_FEATURE_STORE = pmt.DEFAULT_FEATURE_STORE
latest_player_rows = pmt.latest_player_rows
load_field = pmt.load_field
train_medians_and_scalers = pmt.train_medians_and_scalers


def _concat_tsv(main: Path, supplement: Optional[Path]) -> pd.DataFrame:
    usecols = ["season", "start", "tournament", "name", "position", "total"]
    frames: List[pd.DataFrame] = []
    if main.exists():
        df = pd.read_csv(main, sep="\t")
        frames.append(df[[c for c in usecols if c in df.columns]])
    if supplement and supplement.exists():
        sdf = pd.read_csv(supplement, sep="\t")
        frames.append(sdf[[c for c in usecols if c in sdf.columns]])
    if not frames:
        return pd.DataFrame(columns=usecols)
    comb = pd.concat(frames, ignore_index=True)
    comb["start"] = pd.to_datetime(comb["start"], errors="coerce")
    comb = comb.drop_duplicates(subset=["season", "start", "tournament", "name"], keep="last")
    return comb


def _recent_events(sub: pd.DataFrame, as_of: pd.Timestamp, max_rows: int = 8) -> List[Dict[str, Any]]:
    sub = sub[sub["start"] < as_of].sort_values("start", ascending=False).head(max_rows)
    out: List[Dict[str, Any]] = []
    for _, r in sub.iterrows():
        out.append(
            {
                "start": str(r["start"].date()) if pd.notna(r["start"]) else "",
                "tournament": str(r.get("tournament", "")),
                "position": str(r.get("position", "")),
            }
        )
    return out


def audit_one(
    name: str,
    comb: pd.DataFrame,
    fs: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_cols: Sequence[str],
) -> Dict[str, Any]:
    p_comb = comb[comb["name"] == name].copy()
    p_comb["start"] = pd.to_datetime(p_comb["start"], errors="coerce")

    before = p_comb[p_comb["start"] < as_of]
    last_start = before["start"].max() if len(before) else pd.NaT

    fs_names = fs[fs["name"] == name]
    masters_rows = fs[
        (fs["name"] == name) & (fs["tournament"].str.contains("Masters", case=False, na=False))
    ]
    would_masters_fallback = fs_names.empty and not masters_rows.empty
    completely_absent = fs_names.empty and masters_rows.empty

    fs_before = fs_names.copy()
    fs_before["start"] = pd.to_datetime(fs_before["start"], errors="coerce")
    fs_before = fs_before[fs_before["start"] < as_of]

    live_tournament = None
    live_start = None
    future_rows = 0
    if len(fs_names):
        fs_names_dt = fs_names.copy()
        fs_names_dt["start"] = pd.to_datetime(fs_names_dt["start"], errors="coerce")
        future_rows = int((fs_names_dt["start"] >= as_of).sum())

    if len(fs_before):
        last_row = fs_before.sort_values("start").iloc[-1]
        live_tournament = str(last_row["tournament"])
        live_start = str(last_row["start"].date()) if pd.notna(last_row["start"]) else None

    one = latest_player_rows(fs, [name], feature_cols, as_of=as_of)
    inference_tournament = str(one.iloc[0]["tournament"]) if len(one) else None
    inference_start = (
        str(pd.to_datetime(one.iloc[0]["start"]).date()) if len(one) else None
    )

    flags: List[str] = []
    if would_masters_fallback:
        flags.append("MASTERS_FALLBACK")
    if completely_absent:
        flags.append("NO_FEATURE_STORE")
    if future_rows:
        flags.append(f"ROWS_ON_OR_AFTER_AS_OF_{future_rows}")
    if pd.notna(last_start) and last_start < as_of - pd.Timedelta(days=120):
        flags.append("STALE_TSV_LAST_START_120D")

    return {
        "player": name,
        "tsv_rows_total": int(len(p_comb)),
        "tsv_rows_before_as_of": int(len(before)),
        "tsv_last_start": str(last_start.date()) if pd.notna(last_start) else None,
        "recent_events": _recent_events(p_comb, as_of, 8),
        "feature_store_rows_total": int(len(fs_names)),
        "feature_store_rows_before_as_of": int(len(fs_before)),
        "live_row_tournament": live_tournament,
        "live_row_start": live_start,
        "inference_row_tournament": inference_tournament,
        "inference_row_start": inference_start,
        "would_masters_fallback": would_masters_fallback,
        "completely_absent_from_store": completely_absent,
        "future_row_count_in_store": future_rows,
        "flags": flags,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field-file", type=Path, default=None)
    ap.add_argument("--results-tsv", type=Path, default=DEFAULT_RESULTS_TSV)
    ap.add_argument(
        "--results-supplement",
        type=Path,
        default=DEFAULT_RESULTS_SUPPLEMENT,
        help="Set to a nonexistent path to disable supplement",
    )
    ap.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    ap.add_argument("--as-of", type=str, default="2026-04-09")
    ap.add_argument("--json-out", type=Path, default=None, help="Write full audit JSON")
    ap.add_argument("--max-print-flags", type=int, default=40, help="Print players with flags, up to N")
    args = ap.parse_args()

    as_of = pd.Timestamp(args.as_of)
    sup = args.results_supplement if args.results_supplement.exists() else None
    comb = _concat_tsv(args.results_tsv, sup)

    field = load_field(args.field_file, args.results_tsv)
    if not field:
        print("ERROR: empty field (provide --field-file or Masters rows in TSV).", file=sys.stderr)
        sys.exit(1)

    if not args.feature_store.exists():
        print(f"ERROR: feature store missing: {args.feature_store}", file=sys.stderr)
        sys.exit(1)

    fs = pd.read_csv(args.feature_store)
    _, _, _, feature_cols = train_medians_and_scalers(fs)

    rows: List[Dict[str, Any]] = []
    for name in field:
        rows.append(audit_one(name, comb, fs, as_of, feature_cols))

    flagged = [r for r in rows if r["flags"]]
    tsv_latest = comb["start"].max() if len(comb) else pd.NaT

    print("=" * 100)
    print(
        f"Masters field audit  |  as_of={as_of.date()}  |  field={len(field)} players  "
        f"|  merged TSV latest start={tsv_latest.date() if pd.notna(tsv_latest) else 'n/a'}"
    )
    print("=" * 100)
    print(f"Feature store: {args.feature_store}  ({len(fs)} rows)")
    print(f"Supplement:    {sup or '(disabled)'}")
    print()

    hdr = (
        f"{'Player':<28} {'tsv_last':<12} {'fs_live':<12} {'live_event':<36} "
        f"{'Flags':<30}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        live_s = r["live_row_start"] or "—"
        tsv_s = r["tsv_last_start"] or "—"
        ev = (r["live_row_tournament"] or "")[:34]
        fl = ",".join(r["flags"]) if r["flags"] else ""
        print(f"{r['player']:<28} {tsv_s:<12} {live_s:<12} {ev:<36} {fl:<30}")

    print()
    print(f"Players with flags: {len(flagged)} / {len(field)}")
    for r in flagged[: args.max_print_flags]:
        print(f"  • {r['player']}: {', '.join(r['flags'])}")
    if len(flagged) > args.max_print_flags:
        print(f"  ... and {len(flagged) - args.max_print_flags} more (see JSON)")

    payload: Dict[str, Any] = {
        "as_of": str(as_of.date()),
        "n_players": len(field),
        "merged_tsv_latest_start": str(tsv_latest.date()) if pd.notna(tsv_latest) else None,
        "feature_store_path": str(args.feature_store),
        "players": rows,
        "summary": {
            "masters_fallback_count": sum(1 for r in rows if r["would_masters_fallback"]),
            "absent_from_store_count": sum(1 for r in rows if r["completely_absent_from_store"]),
            "future_rows_any_player": sum(r["future_row_count_in_store"] for r in rows),
        },
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
