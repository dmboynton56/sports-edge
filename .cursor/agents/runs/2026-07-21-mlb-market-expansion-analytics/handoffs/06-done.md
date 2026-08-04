# Packet 06 verdict — run line

The selected random-forest classifier produced 2026 YTD Brier **0.22830**, beating the constant pre-test cover-rate baseline (**0.23019**, delta **-0.00189**). Test AUC was **0.55135**, log loss **0.64886**, accuracy **0.64566**, and ECE-10 **0.02415** across 1,428 games. The separately selected random-forest margin head had MAE **3.62657**; its empirical-residual cover probability calibrated worse (Brier **0.23331**, ECE-10 **0.06683**), so the classifier is the delivered `p_home_cover_15` head. Its same-row correlation with v4 moneyline home-win probability was **0.94475**. Home -1.5 and away +1.5 probabilities mirror to one, with no possible push. ROI is deliberately null because there is no run-line odds source.

Verification: `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest -q data-core/tests/unit/test_mlb_runline_model.py data-core/tests/unit/test_mlb_market_features.py` — **9 passed**.

