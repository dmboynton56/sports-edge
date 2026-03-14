-- View: NBA schedules joined with historical book spreads from raw_nba_odds
-- Use this for model training, backtesting, or any analysis needing book_spread.

CREATE OR REPLACE VIEW `sports_edge_raw.nba_schedules_with_odds` AS
SELECT
  s.game_id,
  s.league,
  s.season,
  s.game_date,
  s.home_team,
  s.away_team,
  s.home_score,
  s.away_score,
  spread_o.line AS book_spread,
  total_o.line AS book_total
FROM `sports_edge_raw.raw_schedules` s
LEFT JOIN `sports_edge_raw.raw_nba_odds` spread_o
  ON s.game_id = spread_o.game_id
  AND spread_o.market = 'spread'
LEFT JOIN `sports_edge_raw.raw_nba_odds` total_o
  ON s.game_id = total_o.game_id
  AND total_o.market = 'total'
WHERE s.league = 'NBA';
