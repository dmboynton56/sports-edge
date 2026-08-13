# Dashboard Changelog

## 2026-07-12 — Dashboard v2 initial release

### Added
- `/nba` and `/nfl` live slate boards (Supabase-backed)
- Game detail pages with feature driver charts
- `/models` registry + evaluation/strategy tables
- Injury-aware daily refresh flags (`--injury-aware`, `--include-explanations`)
- `game_explanations` Supabase table + sync script
- `audit_season_readiness.py` and `dashboard_predeploy_gate.py`
- Vercel deploy config for `web/`

### Navigation
- NBA, NFL, Models, and Results added to main nav

### Ops
- Daily refresh applies injury + explanation migrations
- Post-sync NBA/NFL readiness audits in CI
