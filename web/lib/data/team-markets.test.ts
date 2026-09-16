import { describe, expect, it } from "vitest";

import {
  americanImpliedProbability,
  buildTeamMarketPredictions,
  deriveFreshness,
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
    const now = new Date("2026-09-03T18:00:00Z").getTime();
    const rows = buildTeamMarketPredictions("NFL", [game], [prediction], odds, now);
    expect(rows).toHaveLength(6);

    const homeMoneyline = rows.find((row) => row.market === "moneyline" && row.subject.startsWith("SEA"));
    expect(homeMoneyline?.modelProbability).toBe(0.65);
    expect(homeMoneyline?.impliedProbability).toBeCloseTo(0.5798, 3);
    expect(homeMoneyline?.edge).toBeGreaterThan(0);
    expect(homeMoneyline?.ev).toBeCloseTo(0.0833, 3);
    expect(homeMoneyline?.marketStatus).toBe("research");

    const totals = rows.filter((row) => row.market === "total");
    expect(totals).toHaveLength(2);
    expect(totals.every((row) => row.modelVersion === "unmodeled")).toBe(true);
    expect(totals.every((row) => row.edge == null && row.ev == null)).toBe(true);
  });

  it("marks a current prediction with a stale NFL book snapshot as stale", () => {
    const now = new Date("2026-09-16T18:00:00Z").getTime();
    expect(deriveFreshness("2026-09-16T17:40:00Z", -3, "2026-09-07T18:17:00Z", now, 48)).toBe("stale");
    expect(deriveFreshness("2026-09-16T17:40:00Z", -3, "2026-09-16T12:00:00Z", now, 48)).toBe("fresh");
    expect(deriveFreshness("2026-09-16T17:40:00Z", null, null, now, 48)).toBe("no_odds");
  });

  it("withholds NFL edge and EV when the sportsbook snapshot is stale", () => {
    const now = new Date("2026-09-16T18:00:00Z").getTime();
    const rows = buildTeamMarketPredictions("NFL", [game], [prediction], odds, now);
    const moneyline = rows.find((row) => row.market === "moneyline" && row.subject.startsWith("SEA"));
    expect(moneyline).toMatchObject({
      modelProbability: 0.65,
      marketStatus: "model_only",
      edge: null,
      ev: null,
      kelly: null,
    });
    expect(moneyline?.price).toBe(-150);
  });
});
