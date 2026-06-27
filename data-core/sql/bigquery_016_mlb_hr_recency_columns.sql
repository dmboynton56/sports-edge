-- Add batter HR recency context to the BigQuery MLB home run prediction table.
-- Replace PROJECT_ID before running.

alter table `PROJECT_ID.sports_edge_curated.mlb_home_run_predictions`
add column if not exists games_since_last_hr int64;

alter table `PROJECT_ID.sports_edge_curated.mlb_home_run_predictions`
add column if not exists last_hr_date date;
