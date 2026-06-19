# PGA tournament pipeline — refresh & recent changes

## Registry-driven tournament automation

Current tournament automation is driven by `config/pga_tournaments.yaml` and the
single entry point `scripts/refresh_pga_tournament.py`.

```bash
cd data-core

# Auto-detect the active registry window and refresh the right phase.
python scripts/refresh_pga_tournament.py --baseline-only --skip-odds

# Force a specific event or phase when backfilling/testing.
python scripts/refresh_pga_tournament.py \
  --tournament-key us_open_2026 \
  --force-phase live \
  --baseline-only \
  --skip-odds
```

The orchestrator:

- fetches the configured field if the field file is missing,
- runs pre-tournament predictions once unless `--force-pre` is supplied,
- refreshes ESPN leaderboard data during tournament windows,
- runs `scripts/update_pga_midtournament.py` only for new completed-round states,
- applies the cut only after the configured `cut_after_round`,
- exports both `web/public/data/pga_tournaments/{key}.json` and
  `web/public/data/pga_tournaments/current.json`.

GitHub Actions uses `.github/workflows/pga-tournament-refresh.yml`; the legacy
`player-markets-refresh` workflow defaults PGA off so MLB player-market jobs do
not regenerate a tournament baseline.

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
# Optional tour-strength tuning (for sensitivity sweeps):
#   --liv-round-sg-scale 0.91
#   --strong-field-sg-multiplier 1.25

# 3) Retrain v2 models (needed when feature store rows/distribution change)
.venv/bin/python -m src.models.train_models_v2

# 4) Masters pre-event predictions (CSV + meta JSON)
.venv/bin/python scripts/predict_masters_tournament.py --skip-importance

# Optional: audit field vs TSV + feature store (fallback flags, stale TSV)
.venv/bin/python scripts/audit_masters_field_data.py --json-out notebooks/cache/masters_field_audit.json

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
| `notebooks/cache/pga_feature_store_event_level.csv` | Training / inference feature store (large — **gitignored**; regenerate locally after `build_pga_feature_store`) |
| `notebooks/cache/masters_2026_predictions.csv` | Masters field predictions |
| `notebooks/cache/masters_2026_predictions.meta.json` | Run metadata |
| `web/public/data/pga_masters_dashboard.json` | UI bundle |
