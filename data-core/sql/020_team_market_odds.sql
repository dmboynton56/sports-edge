-- Add outcome identity to the shared featured-market odds snapshots.
-- Existing spread-only rows remain valid with a null selection.

alter table odds_snapshots
  add column if not exists selection text,
  add column if not exists provider_event_id text,
  add column if not exists commence_time_utc timestamptz,
  add column if not exists metadata jsonb not null default '{}'::jsonb;

create index if not exists idx_odds_game_market_selection_snapshot
  on odds_snapshots(game_id, market, selection, snapshot_ts desc);

create index if not exists idx_odds_provider_event
  on odds_snapshots(provider_event_id, snapshot_ts desc)
  where provider_event_id is not null;

comment on column odds_snapshots.selection is
  'Canonical outcome side: home, away, over, or under.';

comment on column odds_snapshots.provider_event_id is
  'Provider event identity used to audit schedule matching.';

comment on column odds_snapshots.commence_time_utc is
  'Provider event start time at the moment this price was captured.';

grant select on odds_snapshots to anon, authenticated;
