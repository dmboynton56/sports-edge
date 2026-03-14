-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.features (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  game_id uuid,
  feature_json jsonb NOT NULL,
  asof_ts timestamp with time zone NOT NULL DEFAULT now(),
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT features_pkey PRIMARY KEY (id),
  CONSTRAINT features_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.games(id)
);
CREATE TABLE public.games (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  league text NOT NULL CHECK (league = ANY (ARRAY['NFL'::text, 'NBA'::text])),
  season integer NOT NULL,
  game_time_utc timestamp with time zone NOT NULL,
  home_team text NOT NULL,
  away_team text NOT NULL,
  created_at timestamp with time zone DEFAULT now(),
  week smallint CHECK (week >= 1 AND week <= 30),
  book_spread numeric,
  home_score integer,
  away_score integer,
  CONSTRAINT games_pkey PRIMARY KEY (id)
);
CREATE TABLE public.mancala_stats (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  p1_wins bigint NOT NULL DEFAULT '0'::bigint,
  p2_wins bigint DEFAULT '0'::bigint,
  ties bigint DEFAULT '0'::bigint,
  CONSTRAINT mancala_stats_pkey PRIMARY KEY (id)
);
CREATE TABLE public.model_predictions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  game_id uuid,
  model_name text NOT NULL,
  model_version text NOT NULL,
  my_spread numeric,
  my_home_win_prob numeric,
  asof_ts timestamp with time zone NOT NULL DEFAULT now(),
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT model_predictions_pkey PRIMARY KEY (id),
  CONSTRAINT model_predictions_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.games(id)
);
CREATE TABLE public.model_runs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  league text,
  started_at timestamp with time zone DEFAULT now(),
  finished_at timestamp with time zone,
  rows_written integer,
  success boolean,
  error_text text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT model_runs_pkey PRIMARY KEY (id)
);
CREATE TABLE public.odds_snapshots (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  game_id uuid,
  book text NOT NULL,
  market text NOT NULL CHECK (market = ANY (ARRAY['spread'::text, 'moneyline'::text, 'total'::text])),
  line numeric,
  price numeric,
  snapshot_ts timestamp with time zone NOT NULL DEFAULT now(),
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT odds_snapshots_pkey PRIMARY KEY (id),
  CONSTRAINT odds_snapshots_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.games(id)
);