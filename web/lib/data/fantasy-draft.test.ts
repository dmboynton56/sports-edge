import { describe, expect, it } from "vitest";

import {
  nextPickForSlot,
  recommendDraftPicks,
  snakeDraftTurn,
} from "@/lib/data/fantasy-draft";
import { DEFAULT_FANTASY_ROSTER, type FantasyProjection } from "@/lib/data/fantasy";

type Row = FantasyProjection & { displayPoints: number };

function row(playerId: string, position: FantasyProjection["position"], points: number, availability = "expected"): Row {
  return {
    player_id: playerId,
    player_name: playerId,
    position,
    team: "DEN",
    season: 2026,
    scope: "preseason",
    week: 0,
    projected_games: 17,
    statline: {},
    points,
    displayPoints: points,
    floor_points: points * 0.8,
    ceiling_points: points * 1.2,
    points_per_game: points / 17,
    confidence: "medium",
    availability,
    explanation: [],
    model_version: "test",
    updated_at: "2026-09-02T00:00:00Z",
  };
}

describe("snake draft state", () => {
  it("reverses the slot order every round", () => {
    expect(snakeDraftTurn(1, 12)).toMatchObject({ round: 1, slot: 1 });
    expect(snakeDraftTurn(12, 12)).toMatchObject({ round: 1, slot: 12 });
    expect(snakeDraftTurn(13, 12)).toMatchObject({ round: 2, slot: 12 });
    expect(snakeDraftTurn(24, 12)).toMatchObject({ round: 2, slot: 1 });
  });

  it("finds the user's next pick from any point in the draft", () => {
    expect(nextPickForSlot(1, 12, 5)).toBe(5);
    expect(nextPickForSlot(6, 12, 5)).toBe(20);
    expect(nextPickForSlot(21, 12, 5)).toBe(29);
  });
});

describe("draft recommendations", () => {
  const rows = [
    row("qb1", "QB", 330), row("qb2", "QB", 320),
    row("rb1", "RB", 300), row("rb2", "RB", 280), row("rb3", "RB", 250),
    row("wr1", "WR", 290), row("wr2", "WR", 270), row("wr3", "WR", 245),
    row("te1", "TE", 220), row("te2", "TE", 160),
    row("k1", "K", 190), row("dst1", "DST", 180),
  ];

  it("excludes drafted and unavailable players", () => {
    const recommendations = recommendDraftPicks(
      [...rows, row("rb-out", "RB", 999, "out")],
      new Set(["rb1"]),
      new Set(),
      { ...DEFAULT_FANTASY_ROSTER, teams: 2 },
      1,
      1,
    );
    expect(recommendations.map((item) => item.row.player_id)).not.toContain("rb1");
    expect(recommendations.map((item) => item.row.player_id)).not.toContain("rb-out");
  });

  it("does not chase a second quarterback before open skill-position starters", () => {
    const recommendations = recommendDraftPicks(
      rows,
      new Set(["qb1"]),
      new Set(["qb1"]),
      { ...DEFAULT_FANTASY_ROSTER, teams: 2 },
      3,
      1,
      1,
    );
    expect(recommendations[0].row.position).not.toBe("QB");
  });

  it("defers kicker and defense in early rounds", () => {
    const recommendations = recommendDraftPicks(
      rows,
      new Set(),
      new Set(),
      { ...DEFAULT_FANTASY_ROSTER, teams: 2 },
      1,
      1,
      5,
    );
    expect(recommendations.map((item) => item.row.position)).not.toContain("K");
    expect(recommendations.map((item) => item.row.position)).not.toContain("DST");
  });
});
