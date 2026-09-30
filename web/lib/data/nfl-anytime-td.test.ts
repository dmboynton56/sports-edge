import { afterEach, describe, expect, it, vi } from "vitest";

import {
  isQualifiedAnytimeTdRow,
  getNflAnytimeTdFeed,
  mapAnytimeTdRow,
  type NflAnytimeTdRow,
} from "@/lib/data/nfl-anytime-td";

const baseRow: NflAnytimeTdRow = {
  id: "row-1",
  game_id: "game-1",
  season: 2026,
  week: 1,
  game_date: "2026-09-13",
  game_time_utc: "2026-09-13T17:00:00Z",
  player_id: "player-1",
  player_name: "Example Runner",
  team: "DEN",
  opponent: "KC",
  position: "RB",
  td_probability: 0.32,
  sample_games: 40,
  model_version: "nfl-anytime-td-v1",
  prediction_ts: "2026-09-08T15:00:00Z",
  quality_flags: [],
  best_book: "draftkings",
  best_book_title: "DraftKings",
  best_price: 250,
  market_probability: 0.2857,
  edge: 0.0343,
  ev: 0.12,
  quarter_kelly: 0.017,
  odds_snapshot_ts: "2026-09-03T15:01:00Z",
  odds_status: "priced",
};

afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); vi.useRealTimers(); });

describe("NFL anytime TD serving guardrails", () => {
  it("maps a qualified row to the shared market contract", () => {
    expect(isQualifiedAnytimeTdRow(baseRow)).toBe(true);
    expect(mapAnytimeTdRow(baseRow)).toMatchObject({
      sport: "NFL",
      market: "anytime_td",
      subject: "Example Runner TD (DEN vs KC)",
      price: null,
      edge: null,
      ev: null,
      marketStatus: "model_only",
    });
  });

  it("withholds role-uncertain and invalid probability rows without requiring a price", () => {
    expect(
      isQualifiedAnytimeTdRow({ ...baseRow, quality_flags: ["secondary_depth_role"] }),
    ).toBe(false);
    expect(isQualifiedAnytimeTdRow({ ...baseRow, best_price: null, odds_status: "missing" })).toBe(true);
    expect(isQualifiedAnytimeTdRow({ ...baseRow, td_probability: 0 })).toBe(false);
  });

  it("publishes qualified touchdown predictions with no book price", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-10T13:00:00Z"));
    vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "https://example.supabase.co");
    vi.stubEnv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "test-key");
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [{ ...baseRow, best_price: null, odds_status: "missing" }],
    });
    vi.stubGlobal("fetch", fetchMock);
    const feed = await getNflAnytimeTdFeed();
    expect(feed.predictions).toHaveLength(1);
    expect(feed.predictions[0]).toMatchObject({ price: null, edge: null, marketStatus: "model_only" });
    expect(new URL(fetchMock.mock.calls[0][0]).searchParams.get("order")).toBe("td_probability.desc");
  });

  it("does not report an old or future-cycle feed as freshly generated", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-10T13:00:00Z"));
    vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "https://example.supabase.co");
    vi.stubEnv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "test-key");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [
        { ...baseRow, prediction_ts: "2026-09-01T15:00:00Z" },
        { ...baseRow, game_date: "2026-09-20" },
      ],
    }));
    const feed = await getNflAnytimeTdFeed();
    expect(feed.predictions).toEqual([]);
    expect(feed.generatedAt).toBeNull();
    expect(feed.gaps.join(" ")).toContain("forecast freshness");
  });
});
