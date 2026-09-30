import { describe, expect, it } from "vitest";

import {
  americanImpliedProbability,
  buildTeamMarketPredictions,
  spreadCoverProbability,
} from "@/lib/data/team-markets";

const game = {
  id: "game-1",
  league: "NFL",
  season: 2026,
  week: 1,
  game_time_utc: "2026-09-10T00:20:00Z",
  game_date: "2026-09-09",
  home_team: "SEA",
  away_team: "NE",
  book_spread: -3.5,
};

const prediction = {
  game_id: "game-1",
  my_spread: -7,
  my_home_win_prob: 0.65,
  model_version: "v1",
  asof_ts: "2026-09-03T03:11:48Z",
};

const snapshot = "2026-09-03T14:24:06Z";
const odds = [
  { game_id: "game-1", book: "draftkings", market: "moneyline", selection: "home", line: null, price: -150, snapshot_ts: snapshot },
  { game_id: "game-1", book: "draftkings", market: "moneyline", selection: "away", line: null, price: 130, snapshot_ts: snapshot },
  { game_id: "game-1", book: "draftkings", market: "spread", selection: "home", line: -3.5, price: -110, snapshot_ts: snapshot },
  { game_id: "game-1", book: "draftkings", market: "spread", selection: "away", line: 3.5, price: -110, snapshot_ts: snapshot },
  { game_id: "game-1", book: "draftkings", market: "total", selection: "over", line: 44.5, price: -105, snapshot_ts: snapshot },
  { game_id: "game-1", book: "draftkings", market: "total", selection: "under", line: 44.5, price: -115, snapshot_ts: snapshot },
];

describe("team market normalization", () => {
  it("converts American prices to raw implied probability", () => {
    expect(americanImpliedProbability(-150)).toBeCloseTo(0.6);
    expect(americanImpliedProbability(130)).toBeCloseTo(100 / 230);
  });

  it("produces complementary research cover probabilities", () => {
    const home = spreadCoverProbability(-7, -3.5, "home", 13.9575);
    const away = spreadCoverProbability(-7, 3.5, "away", 13.9575);
    expect(home).toBeGreaterThan(0.5);
    expect(home + away).toBeCloseTo(1, 5);
  });

  it("publishes priced moneyline/spread rows and keeps unmodeled totals honest", () => {
    const rows = buildTeamMarketPredictions("NFL", [game], [prediction], odds);
    expect(rows).toHaveLength(6);

    const homeMoneyline = rows.find((row) => row.market === "moneyline" && row.subject.startsWith("SEA"));
    expect(homeMoneyline?.modelProbability).toBe(0.65);
    expect(homeMoneyline?.impliedProbability).toBeCloseTo(0.5798, 3);
    expect(homeMoneyline?.edge).toBeGreaterThan(0);
    expect(homeMoneyline?.ev).toBeCloseTo(0.0833, 3);

    const totals = rows.filter((row) => row.market === "total");
    expect(totals).toHaveLength(2);
    expect(totals.every((row) => row.modelVersion === "unmodeled")).toBe(true);
    expect(totals.every((row) => row.edge == null && row.ev == null)).toBe(true);
  });

  it("keeps each team's total prices distinct with no invented model output", () => {
    const teamTotals = ["home", "away"].flatMap((side) => ["over", "under"].map((pick) => ({
      game_id: game.id, book: "fanduel", market: "team_total", selection: `${side}_${pick}`,
      line: side === "home" ? 25.5 : 19.5, price: pick === "over" ? -105 : -115, snapshot_ts: snapshot,
    })));
    const rows = buildTeamMarketPredictions("NFL", [game], [prediction], teamTotals);
    expect(rows.map((row) => row.subject)).toEqual([
      "SEA team total over", "SEA team total under", "NE team total over", "NE team total under",
    ]);
    expect(rows.every((row) => row.modelProbability == null && row.edge == null && row.ev == null && row.kelly == null)).toBe(true);
    expect(rows[0].impliedProbability).toBeCloseTo((105 / 205) / (105 / 205 + 115 / 215));
    expect(rows[0].updatedAt).toBe(snapshot);
  });
});
