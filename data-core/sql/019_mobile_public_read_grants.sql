-- Migration: make the read-only mobile serving contract explicit.
-- The native app never connects with a service role. These grants are limited
-- to the tables/views consumed by the server-side mobile API facade.

grant select on games to anon, authenticated;
grant select on odds_snapshots to anon, authenticated;
grant select on model_predictions to anon, authenticated;
grant select on games_today_enriched to anon, authenticated;

grant select on model_evaluation_runs to anon, authenticated;
grant select on strategy_backtest_results to anon, authenticated;
grant select on model_evaluation_history to anon, authenticated;
grant select on game_explanations to anon, authenticated;
grant select on player_availability_reports to anon, authenticated;
grant select on player_impact_estimates to anon, authenticated;

grant select on pga_tournaments to anon, authenticated;
grant select on pga_player_predictions to anon, authenticated;
grant select on pga_odds_snapshots to anon, authenticated;
grant select on pga_player_predictions_latest to anon, authenticated;
grant select on mlb_home_run_predictions to anon, authenticated;
grant select on mlb_home_run_predictions_latest to anon, authenticated;
grant select on mlb_home_run_edges_latest to anon, authenticated;
grant select on mlb_home_run_odds_snapshots to anon, authenticated;
grant select on game_prediction_results to anon, authenticated;
grant select on mlb_home_run_results to anon, authenticated;
grant select on pga_prediction_results to anon, authenticated;

comment on schema public is 'Sports Edge public serving contract is read-only for anon; writes remain pipeline-owned.';
