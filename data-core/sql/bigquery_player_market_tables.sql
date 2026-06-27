-- BigQuery warehouse tables for Sports Edge player-market prediction surfaces.
-- Replace PROJECT_ID before running, or use scripts/sync_player_markets_to_bigquery.py
-- to create these tables automatically.

create schema if not exists `PROJECT_ID.sports_edge_curated`;

create table if not exists `PROJECT_ID.sports_edge_curated.pga_tournaments` (
  event_key string not null,
  season int64 not null,
  name string not null,
  start_date date not null,
  end_date date not null,
  course string,
  par int64,
  field_size int64,
  status string,
  source string,
  raw_record string,
  updated_at timestamp not null
)
partition by start_date
cluster by event_key, season;

create table if not exists `PROJECT_ID.sports_edge_curated.pga_player_predictions` (
  event_key string not null,
  player_name string not null,
  player_id string,
  exp_sg_per_round float64,
  make_cut_prob float64,
  top5_prob float64,
  top10_prob float64,
  top20_prob float64,
  win_prob float64,
  projected_total_strokes float64,
  projected_score_to_par float64,
  model_version string not null,
  prediction_ts timestamp not null,
  simulation_count int64,
  confidence float64,
  quality_flags string,
  feature_snapshot string
)
partition by date(prediction_ts)
cluster by event_key, model_version;

create table if not exists `PROJECT_ID.sports_edge_curated.mlb_home_run_predictions` (
  game_id string not null,
  game_date date not null,
  event_time timestamp,
  player_id string not null,
  player_name string not null,
  team string,
  opponent string,
  venue string,
  lineup_slot int64,
  lineup_status string,
  opposing_probable_pitcher string,
  hr_probability float64 not null,
  baseline_probability float64,
  rank int64,
  games_since_last_hr int64,
  last_hr_date date,
  confidence float64,
  model_version string not null,
  prediction_ts timestamp not null,
  quality_flags string,
  top_features string
)
partition by game_date
cluster by model_version, team;
