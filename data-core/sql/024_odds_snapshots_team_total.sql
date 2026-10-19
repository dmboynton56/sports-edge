-- Allow NFL team_total rows on the shared featured-market odds snapshots.
-- 001_initial_schema.sql constrained market to spread/moneyline/total;
-- 020_team_market_odds.sql added selection identity but left that check in place.

alter table odds_snapshots
  drop constraint if exists odds_snapshots_market_check;

alter table odds_snapshots
  add constraint odds_snapshots_market_check
  check (market in ('spread', 'moneyline', 'total', 'team_total'));

comment on column odds_snapshots.market is
  'Featured market: spread, moneyline, total, or NFL team_total.';
