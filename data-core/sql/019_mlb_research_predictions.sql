-- MLB Research Markets Serving Tables
--
-- This schema supports Research-labeled boards for MLB moneyline (v3), run-line
-- (v1), and totals (v1). These predictions are fail-closed (model-only when no
-- sportsbook prices exist) and clearly labeled as Research, not Trusted.
--
-- Design principles:
-- 1. Single unified table for all research markets (moneyline, run-line, totals)
-- 2. Immutable pregame snapshot (as_of_ts tracks prediction generation)
-- 3. Explicit model_status='research' to distinguish from Trusted/Live boards
-- 4. Fail-closed contract: odds_status tracks whether books are present
-- 5. Compatible with existing BigQuery -> Supabase sync patterns

CREATE TABLE IF NOT EXISTS mlb_research_predictions (
    prediction_id TEXT PRIMARY KEY,
    league TEXT NOT NULL DEFAULT 'MLB',
    market TEXT NOT NULL,  -- 'moneyline', 'run_line', 'total'
    model_version TEXT NOT NULL,  -- 'v3', 'v1'
    model_status TEXT NOT NULL DEFAULT 'research',  -- 'research' or 'trusted'
    
    -- Game identity
    game_id TEXT NOT NULL,
    game_pk BIGINT NOT NULL,
    season INTEGER NOT NULL,
    game_date DATE NOT NULL,
    game_datetime TIMESTAMPTZ,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    venue TEXT,
    
    -- Prediction snapshot
    as_of_ts TIMESTAMPTZ NOT NULL,
    
    -- Moneyline fields
    home_win_prob DOUBLE PRECISION,
    away_win_prob DOUBLE PRECISION,
    
    -- Run-line fields (home -1.5)
    p_home_cover_15 DOUBLE PRECISION,
    p_away_cover_plus_15 DOUBLE PRECISION,
    
    -- Totals fields
    predicted_total DOUBLE PRECISION,
    p_over_8_5 DOUBLE PRECISION,
    p_over_9_5 DOUBLE PRECISION,
    
    -- Odds integration (fail-closed)
    odds_status TEXT NOT NULL DEFAULT 'missing_odds',  -- 'ok', 'missing_odds', 'stale'
    odds_snapshot_ts TIMESTAMPTZ,
    best_book TEXT,
    
    -- Moneyline odds
    home_price DOUBLE PRECISION,  -- American odds
    away_price DOUBLE PRECISION,
    
    -- Run-line odds
    home_runline_price DOUBLE PRECISION,  -- home -1.5 price
    away_runline_price DOUBLE PRECISION,  -- away +1.5 price
    
    -- Totals odds
    total_line DOUBLE PRECISION,  -- e.g., 8.5, 9.5
    over_price DOUBLE PRECISION,
    under_price DOUBLE PRECISION,
    
    -- Edge calculations (NULL when odds_status != 'ok')
    implied_probability DOUBLE PRECISION,
    no_vig_probability DOUBLE PRECISION,
    edge DOUBLE PRECISION,
    ev DOUBLE PRECISION,
    kelly DOUBLE PRECISION,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_market CHECK (market IN ('moneyline', 'run_line', 'total')),
    CONSTRAINT chk_model_status CHECK (model_status IN ('research', 'trusted')),
    CONSTRAINT chk_odds_status CHECK (odds_status IN ('ok', 'missing_odds', 'stale'))
);

-- Indexes for serving queries
CREATE INDEX IF NOT EXISTS idx_mlb_research_predictions_game_date_market 
    ON mlb_research_predictions (game_date DESC, market, model_status);

CREATE INDEX IF NOT EXISTS idx_mlb_research_predictions_game_pk 
    ON mlb_research_predictions (game_pk);

CREATE INDEX IF NOT EXISTS idx_mlb_research_predictions_as_of_ts 
    ON mlb_research_predictions (as_of_ts DESC);

-- View: Latest research predictions per market per game
CREATE OR REPLACE VIEW mlb_research_predictions_latest AS
SELECT DISTINCT ON (game_pk, market)
    prediction_id,
    league,
    market,
    model_version,
    model_status,
    game_id,
    game_pk,
    season,
    game_date,
    game_datetime,
    home_team,
    away_team,
    venue,
    as_of_ts,
    home_win_prob,
    away_win_prob,
    p_home_cover_15,
    p_away_cover_plus_15,
    predicted_total,
    p_over_8_5,
    p_over_9_5,
    odds_status,
    odds_snapshot_ts,
    best_book,
    home_price,
    away_price,
    home_runline_price,
    away_runline_price,
    total_line,
    over_price,
    under_price,
    implied_probability,
    no_vig_probability,
    edge,
    ev,
    kelly,
    created_at,
    updated_at
FROM mlb_research_predictions
ORDER BY game_pk, market, as_of_ts DESC;

-- Grant read access to anon and authenticated roles
GRANT SELECT ON mlb_research_predictions TO anon, authenticated;
GRANT SELECT ON mlb_research_predictions_latest TO anon, authenticated;

-- Comments
COMMENT ON TABLE mlb_research_predictions IS 'Research-labeled MLB predictions for moneyline v3, run-line v1, and totals v1. Fail-closed design: model-only rows when sportsbook prices are unavailable.';
COMMENT ON COLUMN mlb_research_predictions.market IS 'Market type: moneyline, run_line, or total';
COMMENT ON COLUMN mlb_research_predictions.model_status IS 'Always research for these boards; trusted is reserved for production-validated markets like MLB HR';
COMMENT ON COLUMN mlb_research_predictions.odds_status IS 'ok = sportsbook prices present; missing_odds = model-only row; stale = prices too old';
COMMENT ON COLUMN mlb_research_predictions.as_of_ts IS 'Immutable pregame snapshot timestamp; predictions do not update after generation';
