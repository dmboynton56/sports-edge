-- BigQuery warehouse tables for the Sports Edge World Cup prediction pipeline.
-- These tables are the source of truth. Supabase world_cup_* tables are a
-- smaller read cache for the portfolio UI.

CREATE TABLE IF NOT EXISTS `sports_edge_raw.raw_wc_fixtures` (
  external_match_id STRING OPTIONS(description="Source fixture/match ID"),
  season INT64 OPTIONS(description="Tournament year"),
  tournament STRING OPTIONS(description="Tournament name"),
  stage STRING OPTIONS(description="group, round_of_32, round_of_16, quarterfinal, semifinal, final"),
  group_name STRING OPTIONS(description="Group letter/name when applicable"),
  kickoff_utc TIMESTAMP OPTIONS(description="Scheduled kickoff in UTC"),
  home_team STRING OPTIONS(description="Source home/designated team"),
  away_team STRING OPTIONS(description="Source away/designated team"),
  venue STRING OPTIONS(description="Venue or stadium name"),
  status STRING OPTIONS(description="scheduled, in_progress, final, postponed, etc."),
  home_score INT64 OPTIONS(description="Final/current home score"),
  away_score INT64 OPTIONS(description="Final/current away score"),
  neutral_site BOOL OPTIONS(description="Whether the fixture is at a neutral site"),
  source STRING OPTIONS(description="Source system, e.g. fifa, espn_scoreboard"),
  ingested_at TIMESTAMP OPTIONS(description="Warehouse ingestion timestamp"),
  raw_record STRING OPTIONS(description="Raw source JSON")
)
PARTITION BY DATE(kickoff_utc)
CLUSTER BY season, stage, group_name;

CREATE TABLE IF NOT EXISTS `sports_edge_raw.raw_wc_results` (
  match_id STRING,
  match_date DATE,
  season INT64,
  stage STRING,
  home_team STRING,
  away_team STRING,
  home_score INT64,
  away_score INT64,
  source STRING,
  ingested_at TIMESTAMP,
  raw_record STRING
)
PARTITION BY match_date
CLUSTER BY season, stage;

CREATE TABLE IF NOT EXISTS `sports_edge_raw.raw_fifa_rankings` (
  ranking_date DATE,
  team STRING,
  fifa_rank INT64,
  fifa_points FLOAT64,
  source STRING,
  ingested_at TIMESTAMP,
  raw_record STRING
)
PARTITION BY ranking_date
CLUSTER BY team;

CREATE TABLE IF NOT EXISTS `sports_edge_raw.raw_world_elo` (
  snapshot_date DATE,
  team STRING,
  elo FLOAT64,
  elo_rank INT64,
  source STRING,
  ingested_at TIMESTAMP,
  raw_record STRING
)
PARTITION BY snapshot_date
CLUSTER BY team;

CREATE TABLE IF NOT EXISTS `sports_edge_raw.raw_wc_squads` (
  snapshot_date DATE,
  team STRING,
  player_id STRING,
  player_name STRING,
  position STRING,
  club STRING,
  age FLOAT64,
  source STRING,
  ingested_at TIMESTAMP,
  raw_record STRING
)
PARTITION BY snapshot_date
CLUSTER BY team, position;

CREATE TABLE IF NOT EXISTS `sports_edge_raw.raw_wc_player_form` (
  snapshot_date DATE,
  team STRING,
  player_id STRING,
  player_name STRING,
  club STRING,
  minutes FLOAT64,
  goals FLOAT64,
  assists FLOAT64,
  xg FLOAT64,
  xa FLOAT64,
  rating FLOAT64,
  market_value FLOAT64,
  availability FLOAT64,
  injury_status STRING,
  source STRING,
  ingested_at TIMESTAMP,
  raw_record STRING
)
PARTITION BY snapshot_date
CLUSTER BY team;

CREATE TABLE IF NOT EXISTS `sports_edge_raw.raw_wc_odds` (
  snapshot_ts TIMESTAMP,
  market_type STRING OPTIONS(description="match_1x2, futures_winner, etc."),
  match_id STRING,
  team STRING,
  sportsbook STRING,
  decimal_odds FLOAT64,
  american_odds INT64,
  implied_prob FLOAT64,
  source STRING,
  raw_record STRING
)
PARTITION BY DATE(snapshot_ts)
CLUSTER BY market_type, team;

CREATE TABLE IF NOT EXISTS `sports_edge_curated.wc_team_ratings` (
  rating_ts TIMESTAMP,
  season INT64,
  team STRING,
  group_name STRING,
  fifa_rank FLOAT64,
  elo FLOAT64,
  form_points_per_game FLOAT64,
  form_goal_diff_per_game FLOAT64,
  world_cup_experience_score FLOAT64,
  star_player_score FLOAT64,
  host_boost FLOAT64,
  market_rating FLOAT64,
  model_version STRING,
  source_hash STRING
)
PARTITION BY DATE(rating_ts)
CLUSTER BY season, group_name, team;

CREATE TABLE IF NOT EXISTS `sports_edge_curated.wc_feature_snapshots` (
  match_id STRING,
  feature_ts TIMESTAMP,
  season INT64,
  stage STRING,
  group_name STRING,
  home_team STRING,
  away_team STRING,
  feature_json STRING,
  model_version STRING,
  source_hash STRING
)
PARTITION BY DATE(feature_ts)
CLUSTER BY season, stage, group_name;

CREATE TABLE IF NOT EXISTS `sports_edge_curated.wc_match_predictions` (
  prediction_id STRING,
  match_id STRING,
  prediction_ts TIMESTAMP,
  season INT64,
  stage STRING,
  group_name STRING,
  home_team STRING,
  away_team STRING,
  model_name STRING,
  model_version STRING,
  home_win_prob FLOAT64,
  draw_prob FLOAT64,
  away_win_prob FLOAT64,
  home_knockout_win_prob FLOAT64,
  away_knockout_win_prob FLOAT64,
  projected_home_goals FLOAT64,
  projected_away_goals FLOAT64,
  feature_json STRING
)
PARTITION BY DATE(prediction_ts)
CLUSTER BY season, model_version, stage;

CREATE TABLE IF NOT EXISTS `sports_edge_curated.wc_team_probabilities` (
  simulation_ts TIMESTAMP,
  season INT64,
  team STRING,
  group_name STRING,
  model_name STRING,
  model_version STRING,
  simulations INT64,
  bracket_source STRING,
  rating FLOAT64,
  group_prob FLOAT64,
  round_of_32_prob FLOAT64,
  round_of_16_prob FLOAT64,
  quarterfinal_prob FLOAT64,
  semifinal_prob FLOAT64,
  final_prob FLOAT64,
  champion_prob FLOAT64,
  group_rank_probs STRING
)
PARTITION BY DATE(simulation_ts)
CLUSTER BY season, model_version, team;

CREATE TABLE IF NOT EXISTS `sports_edge_curated.wc_simulation_runs` (
  run_id STRING,
  started_at TIMESTAMP,
  finished_at TIMESTAMP,
  season INT64,
  model_name STRING,
  model_version STRING,
  simulations INT64,
  bracket_source STRING,
  rows_written INT64,
  success BOOL,
  error_text STRING
)
PARTITION BY DATE(started_at)
CLUSTER BY season, model_version;
