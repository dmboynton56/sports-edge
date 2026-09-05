import { expect, it } from "vitest";
import { summarizeGameResults, summarizeMlbHr, summarizePga, type GameResultRow, type MlbHrResultRow, type PgaResultRow } from "./results";

it("keeps priced, unpriced, void and ungraded HR outcomes separate", () => {
  const base: MlbHrResultRow = { board_row_id: null, game_date: "2026-09-01", player_name: "Player", team: "A", model_version: "v1", rank: 1, top_k_bucket: "top_10", model_probability: 0.3, actual_home_run: true, actual_plate_appearances: 4, american_price: 200, odds_status: "ok" };
  const rows = [
    base,
    { ...base, actual_home_run: false, american_price: -150 },
    { ...base, odds_status: "missing_odds" },
    { ...base, actual_home_run: false, odds_status: "stale" },
    { ...base, actual_plate_appearances: 0 },
    { ...base, actual_home_run: null },
    { ...base, model_version: "v2", american_price: -200 },
  ];
  const original = structuredClone(rows);
  const [v1, v2] = summarizeMlbHr(rows);
  expect(v1).toMatchObject({ sample: 4, wins: 2, losses: 2, pricedSample: 2, flatUnits: 1, roi: 0.5, modelOnlySample: 2, modelOnlyHitRate: 0.5 });
  expect(v2).toMatchObject({ modelVersion: "v2", pricedSample: 1, flatUnits: 0.5, roi: 0.5 });
  expect(rows).toEqual(original);
});

it("groups team results by league and model while preserving pushes", () => {
  const base: GameResultRow = { league: "NFL", season: 2026, week: 1, game_date: "2026-09-01", home_team: "A", away_team: "B", home_score: 20, away_score: 10, book_spread: -3, my_spread: -4, my_home_win_prob: 0.6, model_version: "v1", spread_result: "win", winner_result: "win", flat_ats_units: 1 };
  const result = summarizeGameResults([base, { ...base, spread_result: "push", winner_result: "loss", flat_ats_units: 0 }, { ...base, league: "NBA", spread_result: "loss", flat_ats_units: -1 }]);
  expect(result).toHaveLength(4);
  expect(result[0]).toMatchObject({ league: "NFL", market: "spread", sample: 2, wins: 1, pushes: 1, roi: 1 });
  expect(result[1]).toMatchObject({ league: "NFL", market: "winner", hitRate: 0.5, roi: null });
  expect(result[2]).toMatchObject({ league: "NBA", losses: 1, roi: -1 });
});

it("keeps PGA models separate and excludes ungraded outcomes", () => {
  const base: PgaResultRow = { event_key: "event", season: 2026, player_name: "Player", model_version: "v1", win_prob: 0.1, top10_prob: 0.4, top20_prob: 0.7, final_position: "1", final_position_numeric: 1, top10_hit: true, top20_hit: true, winner_hit: true, evaluated_at: "2026-09-01" };
  const result = summarizePga([base, { ...base, top10_hit: null, winner_hit: null }, { ...base, model_version: "v2", top10_hit: false, winner_hit: false }]);
  expect(result.map(({ modelVersion, sample, wins, losses }) => ({ modelVersion, sample, wins, losses }))).toEqual([
    { modelVersion: "v1", sample: 1, wins: 1, losses: 0 },
    { modelVersion: "v1", sample: 1, wins: 1, losses: 0 },
    { modelVersion: "v2", sample: 1, wins: 0, losses: 1 },
    { modelVersion: "v2", sample: 1, wins: 0, losses: 1 },
  ]);
});
