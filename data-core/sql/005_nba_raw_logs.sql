-- SQL to create raw_nba_game_logs table in BigQuery
-- Dataset: sports_edge_raw

CREATE TABLE IF NOT EXISTS `sports_edge_raw.raw_nba_game_logs` (
    game_id STRING OPTIONS(description="Unique NBA Game ID"),
    game_date DATE OPTIONS(description="Date the game was played"),
    team STRING OPTIONS(description="Team abbreviation (e.g., GSW)"),
    team_id INT64 OPTIONS(description="NBA official team ID"),
    season INT64 OPTIONS(description="Season starting year (e.g., 2025)"),
    points_scored INT64 OPTIONS(description="Points scored by the team"),
    points_allowed INT64 OPTIONS(description="Points scored by the opponent"),
    net_rating FLOAT64 OPTIONS(description="Point differential (per game proxy)"),
    point_diff FLOAT64 OPTIONS(description="Points scored - Points allowed"),
    ingested_at TIMESTAMP OPTIONS(description="Timestamp of record ingestion"),
    raw_record STRING OPTIONS(description="Full JSON representation of the original API response")
)
PARTITION BY game_date
CLUSTER BY team, season;

