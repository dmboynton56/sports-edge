# PGA Masters / feature pipeline — refresh & recent changes

## Refresh pipeline (after new tour weeks or model tweaks)

Run from the **`data-core`** directory (with `.venv` activated if you use it):

```bash
cd data-core

# 1) Pull completed events from ESPN → supplement TSV
.venv/bin/python scripts/fetch_espn_pga_results.py --season 2026

# 2) Rebuild event-level feature store (merges main archive + supplement)
.venv/bin/python -m src.data.build_pga_feature_store
# Optional: --no-supplement to ignore ESPN supplement
# Optional: --weather-join inner for the old ~41k notebook-style slice only

# 3) Retrain v2 models (needed when feature store rows/distribution change)
.venv/bin/python -m src.models.train_models_v2

# 4) Masters pre-event predictions (CSV + meta JSON)
.venv/bin/python scripts/predict_masters_tournament.py --skip-importance

# 5) Export JSON for the web UI
.venv/bin/python scripts/export_pga_dashboard.py
```

Then serve the Next app from **`web/`** (`npm run dev`) and open **`/pga`**.

---

## What we added / changed

| Area | Description |
|------|-------------|
| **ESPN ingest** | `src/data/espn_pga_results.py` — calendar + `scoreboard?dates=YYYYMMDD` probing, parse R1–R4, positions/ties, CUT/WD/DQ. |
| **Fetch CLI** | `scripts/fetch_espn_pga_results.py` — writes `src/data/archive/pga_results_espn_supplement.tsv`. |
| **Feature store** | `build_pga_feature_store.py` — optional **left** merge of supplement; dedupe on `season,start,tournament,name`. CLI `--no-supplement`. |
| **Masters predict** | `scripts/predict_masters_tournament.py` — uses main + supplement for recent starts and `latest_result_start` in meta. |
| **Web data** | `scripts/export_pga_dashboard.py` → `web/public/data/pga_masters_dashboard.json`. |
| **Web UI** | `web/app/pga/page.tsx` — predictions table, 2026 ESPN events, per-player recent form. Nav already links **PGA** → `/pga`. |

Earlier related work (same initiative): full-archive **`left` weather join** + imputed par/wind so 2023–2025 stay in the store; `predict_masters_tournament.py` and retrained v2 weights on the expanded store.

---

## Known limitations

- **The Sentry** and some events may show **no leaderboard** in ESPN’s API for probed dates; those are skipped.
- **Same-week dual events** (e.g. Puerto Rico Open vs Arnold Palmer): one event may be skipped if `dates=` resolves to the other tournament.
- **Retraining** (step 3) is recommended after supplement/appends change the feature store materially; otherwise scalers/trees can be misaligned.

---

## File paths (quick reference)

| File | Role |
|------|------|
| `src/data/archive/pga_results_2001-2025.tsv` | Primary historical results |
| `src/data/archive/pga_results_espn_supplement.tsv` | ESPN-pulled rows (e.g. 2026) |
| `notebooks/cache/pga_feature_store_event_level.csv` | Training / inference feature store (large — regenerate locally; typically not committed) |
| `notebooks/cache/masters_2026_predictions.csv` | Masters field predictions |
| `notebooks/cache/masters_2026_predictions.meta.json` | Run metadata |
| `web/public/data/pga_masters_dashboard.json` | UI bundle |
