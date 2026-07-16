# Codex task 04 completion

Implemented the Statcast trailing-window refresh and aligned the MLB home-run prediction default slate date with the Denver serving-view convention.

## Files changed

- `.github/workflows/daily-refresh.yml`
  - Added `MLB_HR_STATCAST_REFRESH_DAYS` with a default of 10.
  - Switched the Statcast preload step to the split-window helper while preserving the existing soft-fail warning.
- `data-core/src/models/mlb_hr_statcast_features.py`
  - Added `preload_statcast_cache`, which reads the older interval with `refresh=False`, refreshes only the trailing interval with `refresh=True`, and shares the existing deadline across both calls.
- `data-core/scripts/predict_mlb_home_runs.py`
  - Changed the default slate date from UTC today to the existing `America/Denver` anchor convention. Explicit `--date` behavior is unchanged.
- `data-core/tests/unit/test_mlb_hr_statcast_features.py`
  - Added coverage proving only the trailing 10-day interval is refreshed.
- `data-core/tests/unit/test_plan_daily_refresh.py`
  - Added prediction-side assertions for 2026-07-15 23:30 America/Denver and the mirror 00:30 boundary.

## Verification

- `PYTHONPATH=. .venv/bin/pytest tests/unit/test_mlb_hr_statcast_features.py tests/unit/test_plan_daily_refresh.py -q`
  - `11 passed in 1.42s`
- `PYTHONPATH=. .venv/bin/pytest tests/ -q`
  - `121 passed in 4.80s`
  - Six warnings are scikit-learn `InconsistentVersionWarning` messages from existing serialized test artifacts.
- `.venv/bin/python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('../.github/workflows/daily-refresh.yml').read_text()); print('yaml ok')"`
  - `yaml ok`
- `git diff --check`
  - Passed with no output.
- Protected workflow block comparison against `HEAD`
  - Discord success/failure notification steps are byte-identical.
  - The portfolio repository dispatch block containing `sports-edge-refresh` is byte-identical.

No commits or pushes were made.
