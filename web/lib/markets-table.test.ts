import { describe, expect, it } from "vitest";

import type { Prediction } from "@/lib/data/types";
import {
  DEFAULT_MARKET_TABLE_STATE,
  filterAndSortMarketRows,
  MARKET_TABLE_SORT_KEYS,
  paginateMarketRows,
  readMarketTableState,
  updateMarketTableQuery,
} from "@/lib/markets-table";

function prediction(id: string, overrides: Partial<Prediction> = {}): Prediction {
  return {
    id,
    sport: "NFL",
    league: "NFL",
    gameId: id,
    eventTime: "2026-09-10T18:00:00Z",
    subject: id,
    market: "spread",
    book: "Book A",
    line: -2.5,
    price: -110,
    modelProbability: 0.6,
    impliedProbability: 0.52,
    edge: 0.08,
    ev: 0.1,
    kelly: 0.02,
    confidence: 0.5,
    modelVersion: "v1",
    marketStatus: "research",
    ...overrides,
  };
}

describe("market table state", () => {
  it.each([
    ["all", 3],
    ["10", 3],
    ["25", 2],
    ["50", 1],
    ["75", 0],
  ] as const)("applies the %s probability threshold", (probability, expected) => {
    const rows = [
      prediction("low", { modelProbability: 0.1 }),
      prediction("mid", { modelProbability: 0.25 }),
      prediction("high", { modelProbability: 0.5 }),
    ];
    expect(filterAndSortMarketRows(rows, { ...DEFAULT_MARKET_TABLE_STATE, probability })).toHaveLength(expected);
  });

  it("combines sport, market, book, probability, and status filters", () => {
    const rows = [
      prediction("match", { marketStatus: "supported" }),
      prediction("sport", { sport: "NBA", marketStatus: "supported" }),
      prediction("market", { market: "moneyline", marketStatus: "supported" }),
      prediction("book", { book: "Book B", marketStatus: "supported" }),
      prediction("probability", { modelProbability: 0.49, marketStatus: "supported" }),
      prediction("status", { marketStatus: "research" }),
    ];
    const state = {
      ...DEFAULT_MARKET_TABLE_STATE,
      sport: "NFL",
      market: "spread",
      book: "Book A",
      probability: "50" as const,
      status: "supported" as const,
    };
    expect(filterAndSortMarketRows(rows, state).map((row) => row.id)).toEqual(["match"]);
  });

  it("supports every visible sort key and keeps missing values last in both directions", () => {
    const low = prediction("a", { subject: "A", eventTime: "2026-09-09T18:00:00Z", market: "moneyline", price: -120, modelProbability: 0.4, edge: -0.02, ev: -0.05, marketStatus: "model_only" });
    const high = prediction("b", { subject: "B", eventTime: "2026-09-11T18:00:00Z", market: "spread", price: 120, modelProbability: 0.7, edge: 0.1, ev: 0.15, marketStatus: "supported" });
    const missing = prediction("missing", { eventTime: null, price: null, modelProbability: null, edge: null, ev: null });

    for (const sort of MARKET_TABLE_SORT_KEYS) {
      const ascending = filterAndSortMarketRows([high, low], { ...DEFAULT_MARKET_TABLE_STATE, sort, dir: "asc" });
      const descending = filterAndSortMarketRows([low, high], { ...DEFAULT_MARKET_TABLE_STATE, sort, dir: "desc" });
      expect(ascending.map((row) => row.id)).toEqual(["a", "b"]);
      expect(descending.map((row) => row.id)).toEqual(["b", "a"]);
    }

    for (const sort of ["eventTime", "price", "modelProbability", "edge", "ev"] as const) {
      const ascending = filterAndSortMarketRows([missing, low], { ...DEFAULT_MARKET_TABLE_STATE, sort, dir: "asc" });
      const descending = filterAndSortMarketRows([missing, low], { ...DEFAULT_MARKET_TABLE_STATE, sort, dir: "desc" });
      expect(ascending.at(-1)?.id).toBe("missing");
      expect(descending.at(-1)?.id).toBe("missing");
    }
  });

  it("parses bookmarkable state and resets pagination when filters change", () => {
    const state = readMarketTableState(new URLSearchParams("sport=NFL&market=spread&probability=50&status=research&sort=eventTime&dir=asc&page=3"));
    expect(state).toMatchObject({ sport: "NFL", market: "spread", probability: "50", status: "research", sort: "eventTime", dir: "asc", page: 3 });

    const query = updateMarketTableQuery(new URLSearchParams("sport=NFL&page=3"), { market: "spread" }, true);
    expect(query).toBe("sport=NFL&market=spread");
  });

  it("paginates in stable 50-row windows", () => {
    const rows = Array.from({ length: 105 }, (_, index) => prediction(String(index)));
    expect(paginateMarketRows(rows, 1)).toHaveLength(50);
    expect(paginateMarketRows(rows, 2)[0]?.id).toBe("50");
    expect(paginateMarketRows(rows, 3)).toHaveLength(5);
  });
});
