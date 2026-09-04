import { describe, expect, it } from "vitest";

import {
  isQualifiedAnytimeTdRow,
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
  prediction_ts: "2026-09-03T15:00:00Z",
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

describe("NFL anytime TD serving guardrails", () => {
  it("maps a qualified row to the shared market contract", () => {
    expect(isQualifiedAnytimeTdRow(baseRow)).toBe(true);
    expect(mapAnytimeTdRow(baseRow)).toMatchObject({
      sport: "NFL",
      market: "anytime_td",
      subject: "Example Runner TD (DEN vs KC)",
      price: 250,
      edge: 0.0343,
    });
  });

  it("withholds role-uncertain and extreme longshot rows", () => {
    expect(
      isQualifiedAnytimeTdRow({ ...baseRow, quality_flags: ["secondary_depth_role"] }),
    ).toBe(false);
    expect(isQualifiedAnytimeTdRow({ ...baseRow, best_price: 1200 })).toBe(false);
  });
});
