Implemented task packet 03 exactly.

Changed:

- [018_model_evaluation_history.sql](/home/dmboynton/projects/sports-edge/data-core/sql/018_model_evaluation_history.sql)
- [sync_evaluation_history_to_supabase.py](/home/dmboynton/projects/sports-edge/data-core/scripts/sync_evaluation_history_to_supabase.py)
- [test_sync_evaluation_history.py](/home/dmboynton/projects/sports-edge/data-core/tests/unit/test_sync_evaluation_history.py)
- [performance-history-refresh.yml](/home/dmboynton/projects/sports-edge/.github/workflows/performance-history-refresh.yml)
- [codex-03-done.md](/home/dmboynton/projects/sports-edge/.cursor/orchestration/logs/codex-03-done.md)

Verification:

- Focused tests: 4 passed
- Full data-core suite: 121 passed, 6 existing sklearn warnings
- SQL sanity, Python compilation, workflow assertions, and `git diff --check`: passed
- Live Supabase application was not run
- No commit or push performed
- No web UI or protected schemas changed by task 03