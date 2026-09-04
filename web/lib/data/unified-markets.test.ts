import { describe, expect, it } from "vitest";

import type { MlbResearchPrediction } from "@/lib/data/mlb-research";
import {
  deduplicateWarnings,
  mapMlbResearchPrediction,
  prepareUnifiedMarketRows,
} from "@/lib/data/unified-markets";
import type { Prediction } from "@/lib/data/types";

const researchRow: MlbResearchPrediction = {
  id: "row-1",
  market: "run_line",
  modelVersion: "mlb-runline-v1",
  gameId: "MLB_1",
  gamePk: 1,
  gameDate: "2026-09-04",
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
    marketStatus: "research",
    ...overrides,
  };
}

describe("unified markets", () => {
  it("maps priced and unpriced MLB research rows to the right statuses", () => {
    const priced = mapMlbResearchPrediction(researchRow);
    const modelOnly = mapMlbResearchPrediction({
      ...researchRow,
      oddsStatus: "missing_odds",
      recommendedProbability: undefined,
      recommendedSide: undefined,
      pHomeCover15: 0.44,
      pAwayCoverPlus15: 0.56,
    });

    expect(priced).toMatchObject({
      subject: "SD +1.5 run line",
      line: 1.5,
      price: -115,
      marketStatus: "research",
    });
    expect(priced.impliedProbability).toBeCloseTo(0.54);
    expect(modelOnly).toMatchObject({
      subject: "SD +1.5 run line",
      price: null,
      modelProbability: 0.56,
      marketStatus: "model_only",
    });
  });

  it("includes positive, zero, negative-EV, and model-only rows with null EV last", () => {
    const now = new Date("2026-09-04T00:00:00Z").getTime();
    const result = prepareUnifiedMarketRows([
      prediction({ id: "negative", ev: -0.01 }),
      prediction({ id: "positive", ev: 0.12 }),
      prediction({ id: "zero", ev: 0 }),
      prediction({ id: "model", book: "model", price: null, ev: null, marketStatus: "research" }),
    ], now);

    expect(result.predictions.map((row) => row.id)).toEqual(["positive", "zero", "negative", "model"]);
    expect(result.predictions.at(-1)?.marketStatus).toBe("model_only");
  });

  it("deduplicates rows, filters started events, and withholds sportsbook rows without a model probability", () => {
    const now = new Date("2026-09-04T00:00:00Z").getTime();
    const result = prepareUnifiedMarketRows([
      prediction({ id: "kept" }),
      prediction({ id: "kept", subject: "duplicate" }),
      prediction({ id: "started", eventTime: "2026-09-03T23:00:00Z" }),
      prediction({ id: "unmodeled", modelProbability: null }),
    ], now);

    expect(result.predictions.map((row) => row.id)).toEqual(["kept"]);
    expect(result.hiddenStartedOrInvalid).toBe(1);
    expect(result.missingModelProbability).toBe(1);
  });

  it("deduplicates non-empty warnings", () => {
    expect(deduplicateWarnings(["gap", "gap", "", null, "other"])).toEqual(["gap", "other"]);
  });
});
