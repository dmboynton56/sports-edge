-- Migration: add home_score and away_score columns to games table

alter table games
  add column if not exists home_score integer,
  add column if not exists away_score integer;

comment on column games.home_score is 'Final home team score (null if game not completed).';
comment on column games.away_score is 'Final away team score (null if game not completed).';

-- Create index for faster lookups of completed games
create index if not exists idx_games_scores on games(home_score, away_score) where home_score is not null and away_score is not null;

