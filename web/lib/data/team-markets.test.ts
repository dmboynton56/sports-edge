import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  americanImpliedProbability,
  buildTeamMarketPredictions,
  getTeamMarketPredictions,
  spreadCoverProbability,
} from "@/lib/data/team-markets";
import { supabaseRest } from "@/lib/data/supabase";

vi.mock("@/lib/data/supabase", () => ({
  getSupabaseMissingEnv: () => [],
  supabaseRest: vi.fn(),
}));

afterEach(() => {
  vi.useRealTimers();
  vi.resetAllMocks();
});

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
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-03T21:00:00Z"));
  });

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

  it("publishes complementary model win probabilities without odds or invented prices", () => {
    const rows = buildTeamMarketPredictions("NFL", [game], [prediction], []);
    expect(rows.map((row) => [row.subject, row.modelProbability])).toEqual([
      ["SEA moneyline", 0.65], ["NE moneyline", 0.35],
    ]);
    expect(rows.every((row) => row.book === "model" && row.marketStatus === "model_only")).toBe(true);
    expect(rows.every((row) => row.line == null && row.price == null && row.impliedProbability == null && row.edge == null && row.ev == null && row.kelly == null)).toBe(true);
    expect(rows.every((row) => row.updatedAt === prediction.asof_ts)).toBe(true);
  });

  it("fills only the missing moneyline outcome without duplicating the captured side", () => {
    const rows = buildTeamMarketPredictions("NFL", [game], [prediction], [odds[0]]);
    expect(rows).toHaveLength(2);
    expect(new Set(rows.map((row) => row.id)).size).toBe(2);
    expect(rows[0].book).toBe("draftkings");
    expect(rows[1].subject).toBe("NE moneyline");
    expect(rows[1].price).toBeNull();
    expect(rows[1].ev).toBeNull();
  });

  it.each([null, Number.NaN, -0.1, 1.1])("withholds model-only rows for invalid probability %s", (probability) => {
    expect(buildTeamMarketPredictions("NFL", [game], [{ ...prediction, my_home_win_prob: probability }], [])).toEqual([]);
  });

  it("does not fabricate forecasts when the model row is missing", () => {
    expect(buildTeamMarketPredictions("NFL", [game], [], [])).toEqual([]);
  });

  it.each(["", "invalid", "2026-08-30T21:00:00Z", "2026-09-04T21:00:00Z"])("withholds stale or invalid model-only forecasts (%s)", (asofTs) => {
    const rows = buildTeamMarketPredictions("NFL", [game], [{ ...prediction, asof_ts: asofTs }], []);
    expect(rows).toEqual([]);
  });

  it("does not revive an old NBA forecast when sportsbook odds are missing", () => {
    expect(buildTeamMarketPredictions("NBA", [game], [{ ...prediction, asof_ts: "2026-09-02T03:11:48Z" }], [])).toEqual([]);
  });

  it("keeps each team's total prices distinct with no invented model output", () => {
    const teamTotals = ["home", "away"].flatMap((side) => ["over", "under"].map((pick) => ({
      game_id: game.id, book: "fanduel", market: "team_total", selection: `${side}_${pick}`,
      line: side === "home" ? 25.5 : 19.5, price: pick === "over" ? -105 : -115, snapshot_ts: snapshot,
    })));
    const rows = buildTeamMarketPredictions("NFL", [game], [prediction], teamTotals)
      .filter((row) => row.market === "team_total");
    expect(rows.map((row) => row.subject)).toEqual([
      "SEA team total over", "SEA team total under", "NE team total over", "NE team total under",
    ]);
    expect(rows.every((row) => row.modelProbability == null && row.edge == null && row.ev == null && row.kelly == null)).toBe(true);
    expect(rows[0].impliedProbability).toBeCloseTo((105 / 205) / (105 / 205 + 115 / 215));
    expect(rows[0].updatedAt).toBe(snapshot);
  });
});

describe("NFL forecast publication", () => {
  it.each([
    ["2026-09-29T20:00:00Z", 2],
    ["2026-09-27T20:00:00Z", 0],
  ])("publishes only current-cycle model-only forecasts (%s)", async (asofTs, expectedRows) => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-30T21:00:00Z"));
    vi.mocked(supabaseRest).mockImplementation(async (resource) => {
      if (resource.startsWith("games?")) return [{ ...game, game_date: "2026-10-04", game_time_utc: "2026-10-04T17:00:00Z" }];
      if (resource.startsWith("model_predictions?")) return [{ ...prediction, asof_ts: asofTs }];
      return [];
    });

    const feed = await getTeamMarketPredictions("NFL");
    expect(feed.predictions).toHaveLength(expectedRows);
    expect(feed.predictions.every((row) => row.marketStatus === "model_only" && row.ev == null)).toBe(true);
    expect(feed.gaps).toContain("NFL featured-market outcome coverage is incomplete (0/6).");
  });
});
