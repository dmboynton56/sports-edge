-- Migration: enforce stable identities for evaluation and strategy rows.

create unique index if not exists idx_model_eval_unique_identity
  on model_evaluation_runs(league, model_name, model_version, evaluation_name);

create unique index if not exists idx_strategy_backtests_unique_identity
  on strategy_backtest_results(evaluation_run_id, strategy_id);
