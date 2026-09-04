import { describe, expect, it } from "vitest";

import type { MlbResearchPrediction } from "@/lib/data/mlb-research";
import { filterCurrentPositiveEv, mapMlbResearchPrediction } from "@/lib/data/unified-markets";
import type { Prediction } from "@/lib/data/types";

const researchRow: MlbResearchPrediction = {
  id: "row-1",
  market: "run_line",
  modelVersion: "mlb-runline-v1",
  gameId: "MLB_1",
  gamePk: 1,
  gameDate: "2026-09-03",
  eventTime: "2026-09-04T02:00:00Z",
  homeTeam: "COL",
  awayTeam: "SD",
  venue: null,
  oddsStatus: "ok",
  bestBook: "book",
  homeRunlinePrice: -105,
  awayRunlinePrice: -115,
  homeRunlineLine: -1.5,
  edge: 0.04,
  ev: 0.08,
  kelly: 0.02,
  recommendedSide: "away",
  recommendedProbability: 0.58,
  asOfTs: "2026-09-03T23:00:00Z",
};

function prediction(overrides: Partial<Prediction> = {}): Prediction {
  return {
    id: "prediction-1",
    sport: "CFB",
    league: "CFB",
    gameId: "game-1",
    eventTime: "2026-09-04T02:00:00Z",
    subject: "Away spread",
    market: "spread",
    book: "book",
    line: 3.5,
    price: -110,
    modelProbability: 0.56,
    impliedProbability: 0.52,
    edge: 0.04,
    ev: 0.07,
    kelly: 0.01,
    confidence: 0.6,
    modelVersion: "v1",
    ...overrides,
  };
}

describe("unified research markets", () => {
  it("maps the recommended MLB side, price, line, and implied probability", () => {
    const mapped = mapMlbResearchPrediction(researchRow);

    expect(mapped.subject).toBe("SD +1.5 run line");
    expect(mapped.line).toBe(1.5);
    expect(mapped.price).toBe(-115);
    expect(mapped.impliedProbability).toBeCloseTo(0.54);
  });

  it("keeps only future positive-EV rows and sorts highest first", () => {
    const now = new Date("2026-09-04T00:00:00Z").getTime();
    const rows = filterCurrentPositiveEv([
      prediction({ id: "low", ev: 0.03 }),
      prediction({ id: "high", ev: 0.12 }),
      prediction({ id: "started", eventTime: "2026-09-03T23:00:00Z", ev: 0.5 }),
      prediction({ id: "negative", ev: -0.01 }),
      prediction({ id: "unpriced", ev: null }),
    ], now);

    expect(rows.map((row) => row.id)).toEqual(["high", "low"]);
  });
});
