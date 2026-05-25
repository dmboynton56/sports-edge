-- Migration: allow MLB games and add probable-pitcher display fields.

alter table games
  drop constraint if exists games_league_check;

alter table games
  add constraint games_league_check
  check (league in ('NFL', 'NBA', 'MLB'));

alter table games
  add column if not exists home_probable_pitcher text,
  add column if not exists away_probable_pitcher text;

comment on column games.home_probable_pitcher is 'Probable home starter for MLB display, when available.';
comment on column games.away_probable_pitcher is 'Probable away starter for MLB display, when available.';
