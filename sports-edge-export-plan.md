# Sports-Edge: Export & Deployment Plan (Portfolio Integration)

> Goal: Move cleaned features + model outputs from the Python pipeline into the **Supabase** DB that powers the portfolio, with a repeatable deployment path and auditability.

---

## 1) Environment & Secrets
- **Supabase**: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (write), `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` (read).
- **Odds API**: `ODDS_API_KEY`.
- Store secrets:
  - **GitHub Actions** → repository secrets.
  - **Local dev** → `.env`, never commit.

---

## 2) Schema Migration (SQL)
Create migrations in `/supabase/migrations` or keep a `sql/` folder in the analysis repo and run them via CI.

**Tables**
```sql
create table if not exists games (
  id uuid primary key default gen_random_uuid(),
  league text check (league in ('NFL','NBA')) not null,
  season int not null,
  game_time_utc timestamptz not null,
  home_team text not null,
  away_team text not null
);

create table if not exists odds_snapshots (
  id uuid primary key default gen_random_uuid(),
  game_id uuid references games(id) on delete cascade,
  book text not null,
  market text check (market in ('spread','moneyline','total')) not null,
  line numeric,
  price numeric,
  snapshot_ts timestamptz not null default now()
);

create table if not exists model_predictions (
  id uuid primary key default gen_random_uuid(),
  game_id uuid references games(id) on delete cascade,
  model_name text not null,
  model_version text not null,
  my_spread numeric,
  my_home_win_prob numeric,
  asof_ts timestamptz not null default now()
);

create table if not exists features (
  id uuid primary key default gen_random_uuid(),
  game_id uuid references games(id) on delete cascade,
  feature_json jsonb not null,
  asof_ts timestamptz not null default now()
);

create table if not exists model_runs (
  id uuid primary key default gen_random_uuid(),
  league text,
  started_at timestamptz default now(),
  finished_at timestamptz,
  rows_written int,
  success boolean,
  error_text text
);
```

**View (latest odds+pred per game for today)**
```sql
create or replace view games_today_enriched as
select
  g.league, g.season, g.game_time_utc,
  g.home_team, g.away_team,
  o.line as book_spread,
  p.my_spread,
  (p.my_spread - o.line) as edge_pts,
  p.my_home_win_prob,
  p.model_version,
  p.asof_ts
from games g
left join lateral (
  select line
  from odds_snapshots o
  where o.game_id = g.id and o.market = 'spread'
  order by snapshot_ts desc limit 1
) o on true
left join lateral (
  select my_spread, my_home_win_prob, model_version, asof_ts
  from model_predictions p
  where p.game_id = g.id
  order by asof_ts desc limit 1
) p on true
where g.game_time_utc::date = (now() at time zone 'America/Denver')::date
order by g.game_time_utc;
```

**RLS Policies (public read-only)**
```sql
alter table games enable row level security;
alter table odds_snapshots enable row level security;
alter table model_predictions enable row level security;
alter table features enable row level security;
alter table model_runs enable row level security;

create policy "public read games" on games for select using (true);
create policy "public read odds" on odds_snapshots for select using (true);
create policy "public read preds" on model_predictions for select using (true);
create policy "public read features" on features for select using (true);
create policy "public read runs" on model_runs for select using (true);
```

---

## 3) Batch Export Job (Python CLI)
Location: a dedicated repo (e.g., `sports-edge-pipeline/`) or a `/scripts/` folder in your monorepo.

**CLI outline**
```
python -m sports_edge.refresh --league NFL --date 2025-11-06 --runs 1
  # steps:
  # 1) fetch schedule + team form (league-specific)
  # 2) fetch latest odds
  # 3) build features (respect feature contract)
  # 4) run model inference (load artifact)
  # 5) upsert games, odds_snapshots, model_predictions, features
  # 6) write model_runs row (audit)
```

**Idempotency**
- Upsert keyed by `(league, season, home_team, away_team, game_time_utc)` for `games`.
- For odds: `(game_id, book, market, snapshot_ts)`.
- For preds: `(game_id, model_version, asof_ts)`.

---

## 4) Scheduling
Two options (both OK for resume bullet points):
- **Supabase Scheduled Functions** (Edge Functions + `pg_cron`) — low infra, easy secret sharing.
- **GitHub Actions (cron)** — portable: `on: schedule: '*/15 * * * *'`. The job runs Python, writes to Supabase via service key.

---

## 5) CI/CD & Artifacts
- Persist model artifact (`.pkl` or ONNX) with semantic version (e.g., `v0.1.0`) in a release or object storage.
- Every run logs `model_version` into `model_predictions` and `model_runs`.
- Add a small `healthcheck` step to confirm DB connectivity and table existence before writes.

---

## 6) Observability
- Emit run metrics to `model_runs`.
- Optional Slack/Email on failure via GitHub Actions or a Supabase function webhook.
