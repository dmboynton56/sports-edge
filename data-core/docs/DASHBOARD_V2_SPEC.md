# Dashboard v2 Spec — web-First Serving

Generated: 2026-07-12

## Serving contract
- **Canonical surface:** `sports-edge/web` deployed to Vercel
- **Live data:** Supabase REST (`games`, `model_predictions`, `games_today_enriched`, eval tables)
- **Static fallback:** `performance_history.json` only (eval UI); NBA/NFL slates are Supabase-only in v1

## Routes

### `/nba`
- Today's NBA slate from Supabase (extendable date window)
- Columns: matchup, game time, book spread, model spread, edge pts, win prob, model version, freshness badge
- Empty state when off-season or no rows

### `/nfl`
- Current NFL week slate (7-day window, not Denver-today only)
- Same columns as NBA + week number
- Link to game detail

### `/nba/[gameId]`, `/nfl/[gameId]`
- Matchup header, spreads, win prob
- Feature drivers chart (≥5 features)
- Injury badge: `injury_adjusted`, `injury_data_missing`, deltas
- Base vs adjusted comparison when both exist (M4)

### `/models`
- Registry from `data-core/models/DECISIONS.md`
- Live metrics from `model_evaluation_runs` + `strategy_backtest_results`
- Fallback to `performance_history.json`

### `/performance` (enhanced)
- League tabs: NBA, NFL first
- ROI chart + strategy tables

### Nav updates
- Add NBA, NFL, Models, Results
- Markets page links to NBA/NFL boards

## Data adapters

| File | Source |
|------|--------|
| `web/lib/data/team-markets.ts` | `games` + `model_predictions` + `odds_snapshots` |
| `web/lib/data/explanations.ts` | `game_explanations` |
| `web/lib/data/evaluations.ts` | `model_evaluation_runs`, `strategy_backtest_results` |

## API routes
- `GET /api/nba-slate` — JSON slate payload
- `GET /api/nfl-slate` — JSON slate payload

## Freshness badges
| Badge | Condition |
|-------|-----------|
| `fresh` | prediction_ts within 24h |
| `stale` | prediction_ts > 24h |
| `no_odds` | book_spread null |
| `injury_adjusted` | explanation.injury_adjusted true |
| `injury_data_missing` | high-impact game, no impact rows |

## Backend scripts
- `audit_season_readiness.py` — schedule/prediction/odds/dupes/stale/injury audit
- `sync_explanations_to_supabase.py` — persist top features + injury flags
- `018_game_explanations.sql` — explanation table DDL

## Deploy
- Vercel root: `sports-edge/web`
- Required env: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`

## Production gates (from PRODUCTION_ROADMAP)
A model version is not promoted unless evaluation sample, calibration, strategy ROI, serving validation, dedupe, and injury representation all pass.
