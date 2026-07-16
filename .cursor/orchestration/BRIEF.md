# Sports Edge ops dashboard — orchestrated delivery brief

Repo: `/home/dmboynton/projects/sports-edge`
Existing plan (must still land): `.cursor/plans/2026-07-hardening-and-ops-dashboard.md`

## Product intent (user)

Make the ops dashboard the place people look for model numbers.

### Navigation

- Top nav should be lean: Overview, Markets, Performance, Insights, Data Quality (and Results if it earns a slot).
- **Remove PGA and CBB as top-level tabs.** They belong under Markets (and Performance), not as peer top-nav items.
- College basketball can exist in the hierarchy but is low priority / seasonal — don't promote it.

### Markets (primary surface)

Hierarchy:

```
/markets
  /markets/nba/...
  /markets/pga/...
  /markets/mlb/...
  /markets/nfl/...   (scaffold ok if thin)
  /markets/nhl/...   (scaffold ok if thin)
  /markets/cbb/...   (scaffold / de-emphasized)
```

Within a sport, specific markets, e.g.:

- MLB → home runs (existing board), winner/team markets as available
- PGA → top-10 / winner tournament board (move `/pga` under `/markets/pga`)
- NBA/NFL → existing team/ATS prediction surfaces

Someone should click Markets → sport → market and see the numbers the models produced.

### Performance (mirror Markets)

Same sport → market hierarchy. For each market, show backtested / graded performance over selectable windows (e.g. last N days / season / all available).

This requires producing and **persisting** backtest/evaluation numbers (not notebook-only artifacts), so future training/test plans can consume them. Prefer Supabase + existing sync scripts (`export_performance_history.py`, `sync_evaluation_history_to_supabase.py`, player-market grading from the plan) over one-off JSON in git.

### Insights

When backtests produce findings worth keeping, add/update Insights content (e.g. MLB HR pytorch insight page pattern) rather than dumping everything on Overview.

### Hardening plan (still in scope)

Phases 1–4 of `2026-07-hardening-and-ops-dashboard.md` remain goals:

- MLB HR: odds on daily path, Statcast refresh, loud degradation, timezone unify, grading table
- PGA: generic field fetcher, season registry, live reliability, Supabase-first serving, grading
- Dashboard: results-first overview, live grading, automate performance artifacts
- Tests + don't break `games` / `model_predictions` / `odds_snapshots` schemas

UI reorganization above is the Phase 3 UX shape — Markets/Performance sport trees replace dumping PGA/CBB in the top bar.

## Constraints

- Do not break portfolio-consumed Supabase schemas (`games`, `model_predictions`, `odds_snapshots`).
- Keep `sports-edge-refresh` dispatch + Discord notifications intact where workflows are touched.
- New tables/views → `data-core/sql/` next migration number (≥017).
- Prefer matching existing dashboard patterns (`AppShell`, `PageHeader`, `web/lib/data/*`, shadcn).
- No secrets in commits. Don't push until Fable `APPROVE`.

## Deliverables Fable must produce in planning

Write under `.cursor/orchestration/`:

1. `PLAN.md` — ordered work packets, acceptance criteria, what Codex vs Fable owns, risks
2. `CODEX_TASKS/NN-<slug>.md` — each file is a self-contained Codex prompt (context paths, exact files to touch, done definition, out-of-scope)
3. `ACCEPTANCE.md` — checklist Fable will use in review (must include Markets/Performance hierarchy + nav cleanup + persisted backtest/eval numbers + plan hardening items that are in this round)

Planning stance: Fable explores the codebase, designs the cut, **does not implement** during planning. Implementation is Codex's job unless a task is marked Fable-only (rare: schema design judgment calls).
