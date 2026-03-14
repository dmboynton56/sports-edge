-- BQML scoring query: write predictions for upcoming games into curated table.
-- Schedule daily. Parametrize via @project, @model_version, @model_number, @start_date, @end_date.

DECLARE project STRING DEFAULT "@project";
DECLARE model_version STRING DEFAULT "v1";
DECLARE model_number STRING DEFAULT "v1";
DECLARE start_date DATE DEFAULT CURRENT_DATE();
DECLARE end_date DATE DEFAULT DATE_ADD(CURRENT_DATE(), INTERVAL 7 DAY);

INSERT INTO `${project}.sports_edge_curated.model_predictions` (
  prediction_id, game_id, league, season, season_week,
  model_name, model_version, model_number,
  predicted_spread, home_win_prob, prediction_ts, input_hash
)
SELECT
  CONCAT(f.game_id, "_", model_version, "_", FORMAT_TIMESTAMP("%Y%m%dT%H%M%S", CURRENT_TIMESTAMP())) AS prediction_id,
  f.game_id,
  f.league,
  f.season,
  f.week_number AS season_week,
  "bqml_spread" AS model_name,
  model_version,
  model_number,
  -predicted_spread AS predicted_spread,            -- flip to home-team perspective
  1.0 / (1.0 + EXP(-predicted_spread / 7.0)) AS home_win_prob, -- replace with calibrated link if stored
  CURRENT_TIMESTAMP() AS prediction_ts,
  TO_HEX(MD5(TO_JSON_STRING(f))) AS input_hash
FROM
  ML.PREDICT(
    MODEL `${project}.sports_edge_curated.models_spread_${model_version}`,
    (
      SELECT * FROM `${project}.sports_edge_curated.feature_snapshots`
      WHERE game_date BETWEEN start_date AND end_date
        AND league = "NFL"
    )
  ) AS p
JOIN `${project}.sports_edge_curated.feature_snapshots` f
  ON p.game_id = f.game_id
WHERE f.game_date BETWEEN start_date AND end_date;
