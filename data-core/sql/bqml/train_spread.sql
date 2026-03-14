-- BQML training query for spread regression.
-- Schedule weekly (or after feature changes). Parametrize via @project, @model_version, @feature_version.

DECLARE project STRING DEFAULT "@project";
DECLARE model_version STRING DEFAULT "v1";
DECLARE feature_version STRING DEFAULT "v2";

CREATE OR REPLACE MODEL `${project}.sports_edge_curated.models_spread_${model_version}`
OPTIONS(
  model_type = "linear_reg",
  input_label_cols = ["home_margin"],
  l1_reg = 0.0,
  l2_reg = 1.0,
  data_split_method = "AUTO_SPLIT"
) AS
SELECT
  home_margin AS label,
  rest_home, rest_away, b2b_home, b2b_away,
  opp_strength_home_season, opp_strength_away_season,
  home_team_win_pct, away_team_win_pct,
  home_team_point_diff, away_team_point_diff,
  rest_differential, win_pct_differential, point_diff_differential, opp_strength_differential,
  week_number, month, is_playoff,
  form_home_epa_off_3, form_home_epa_off_5, form_home_epa_off_10,
  form_home_epa_def_3, form_home_epa_def_5, form_home_epa_def_10,
  form_away_epa_off_3, form_away_epa_off_5, form_away_epa_off_10,
  form_away_epa_def_3, form_away_epa_def_5, form_away_epa_def_10,
  form_epa_off_diff_3, form_epa_off_diff_5, form_epa_off_diff_10,
  form_epa_def_diff_3, form_epa_def_diff_5, form_epa_def_diff_10
FROM `${project}.sports_edge_curated.training_feature_view_${feature_version}`;
