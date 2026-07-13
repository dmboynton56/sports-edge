# Sports Model Performance History

Generated: 2026-07-13

| Sport | Version | Season | Market | Sample | Primary metrics | Odds status |
| --- | --- | --- | --- | --- | --- | --- |
| NBA | v3 | 2025-26 | spread | 1175 completed; 739 BQ odds; 64 Supabase graded | ATS ROI -7.5%; BQ default ROI -0.5% | oddspapi_tail_patch_partial |
| NFL | v1 | 2025 | spread | 285 BQ scored; 56 Supabase graded | ATS ROI -8.0%; BQ AUC 0.5790; spread MAE 11.15 | partial_supabase_spread_odds |
| MLB | v3 | 2026 YTD | moneyline | 673 test games; 673 odds rows | Brier 0.2478; log loss 0.6888; AUC 0.5431; ROI -3.1% | free_historical_moneylines |
| PGA | v2 artifacts evaluated on refreshed store | test >= 2025 | outright/top placement | 15172 regression rows | SG Spearman 0.391; made-cut Brier 0.200; win AUC 0.756 | masters_2026_pre_event_odds_cache_only |
| CBB | manual matchup artifacts | CV 2016-2025 | tournament winner probability | 9 folds; 2002 matchup rows | XGB mean log loss 0.5751; Brier 0.1979; AUC 0.7581 | no_sportsbook_odds |

## Blocking Gaps

- NBA: raw_nba_odds stale before OddsPapi tail patch for Feb-May 2026
- NBA: Supabase season still has historical rows missing book_spread
- NFL: OddsPapi historical NFL coverage is recent-fixture limited on this key tier
- NFL: Supabase has stale v1/v2/v3 version mix
- NFL: Available ATS sample is not full season
- MLB: Free historical moneylines are consensus/comparison sources, not sharp-book closing lines
- MLB: MLB strategy remains negative ROI on broader free-history backtests
- PGA: No historical event-by-event sportsbook odds ROI history
- PGA: Player name normalization limits post-mortems
- PGA: ESPN ingest misses no-board/same-week collision events
- CBB: Raw Kaggle MMLM files are absent from the repo
- CBB: 2026 tournament labels are not present in the matchup feature store
- CBB: No sportsbook odds or ROI history is available
