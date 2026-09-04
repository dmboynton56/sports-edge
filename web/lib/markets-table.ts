import { isFiniteNumber } from "@/lib/data/json";
import type { Prediction } from "@/lib/data/types";

export const MARKET_TABLE_PAGE_SIZE = 50;

export const MARKET_TABLE_SORT_KEYS = [
  "subject",
  "eventTime",
  "market",
  "price",
  "modelProbability",
  "edge",
  "ev",
  "marketStatus",
] as const;

export type MarketTableSortKey = (typeof MARKET_TABLE_SORT_KEYS)[number];
export type MarketTableSortDirection = "asc" | "desc";
export type MarketTableStatus = "all" | Prediction["marketStatus"];

export type MarketTableState = {
  sport: string;
  market: string;
  book: string;
  probability: "all" | "10" | "25" | "50" | "75";
  status: MarketTableStatus;
  sort: MarketTableSortKey;
  dir: MarketTableSortDirection;
  page: number;
};

export const DEFAULT_MARKET_TABLE_STATE: MarketTableState = {
  sport: "all",
  market: "all",
  book: "all",
  probability: "all",
  status: "all",
  sort: "ev",
  dir: "desc",
  page: 1,
};

function finiteDate(value: string | null): number | null {
  if (!value) return null;
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? timestamp : null;
}

function compareNullableNumbers(
  left: number | null | undefined,
  right: number | null | undefined,
  direction: MarketTableSortDirection,
): number {
  const leftMissing = left == null || !Number.isFinite(left);
  const rightMissing = right == null || !Number.isFinite(right);
  if (leftMissing && rightMissing) return 0;
  if (leftMissing) return 1;
  if (rightMissing) return -1;
  const comparison = left - right;
  return direction === "asc" ? comparison : -comparison;
}

function compareNullableStrings(
  left: string | null | undefined,
  right: string | null | undefined,
  direction: MarketTableSortDirection,
): number {
  if (left == null && right == null) return 0;
  if (left == null) return 1;
  if (right == null) return -1;
  const comparison = left.localeCompare(right);
  return direction === "asc" ? comparison : -comparison;
}

export function compareMarketRows(
  left: Prediction,
  right: Prediction,
  key: MarketTableSortKey,
  direction: MarketTableSortDirection,
): number {
  if (key === "eventTime") {
    return compareNullableNumbers(finiteDate(left.eventTime), finiteDate(right.eventTime), direction);
  }
  if (key === "price" || key === "modelProbability" || key === "edge" || key === "ev") {
    return compareNullableNumbers(left[key], right[key], direction);
  }
  return compareNullableStrings(left[key], right[key], direction);
}

function parseProbability(value: string | null): MarketTableState["probability"] {
  if (value === "10" || value === "25" || value === "50" || value === "75") return value;
  return "all";
}

function parseStatus(value: string | null): MarketTableStatus {
  if (value === "supported" || value === "research" || value === "model_only") return value;
  return "all";
}

function parseSort(value: string | null): MarketTableSortKey {
  switch (value) {
    case "subject":
    case "eventTime":
    case "market":
    case "price":
    case "modelProbability":
    case "edge":
    case "ev":
    case "marketStatus":
      return value;
    default:
      return "ev";
  }
}

export function readMarketTableState(searchParams: URLSearchParams): MarketTableState {
  const dir = searchParams.get("dir") ?? "desc";
  const parsedPage = Number.parseInt(searchParams.get("page") ?? "1", 10);

  return {
    sport: searchParams.get("sport") || "all",
    market: searchParams.get("market") || "all",
    book: searchParams.get("book") || "all",
    probability: parseProbability(searchParams.get("probability")),
    status: parseStatus(searchParams.get("status")),
    sort: parseSort(searchParams.get("sort")),
    dir: dir === "asc" ? "asc" : "desc",
    page: Number.isFinite(parsedPage) && parsedPage > 0 ? parsedPage : 1,
  };
}

export function updateMarketTableQuery(
  current: URLSearchParams,
  updates: Partial<Record<keyof MarketTableState, string | number>>,
  resetPage = false,
): string {
  const next = new URLSearchParams(current.toString());
  const keys = ["sport", "market", "book", "probability", "status", "sort", "dir", "page"] satisfies (keyof MarketTableState)[];
  for (const key of keys) {
    const rawValue = updates[key];
    if (rawValue == null) continue;
    const value = String(rawValue);
    const defaultValue = String(DEFAULT_MARKET_TABLE_STATE[key]);
    if (value === defaultValue) next.delete(key);
    else next.set(key, value);
  }
  if (resetPage && !("page" in updates)) next.delete("page");
  return next.toString();
}

export function filterAndSortMarketRows(
  predictions: Prediction[],
  state: MarketTableState,
): Prediction[] {
  const minimumProbability = state.probability === "all" ? null : Number(state.probability) / 100;
  return predictions
    .filter((prediction) => state.sport === "all" || prediction.sport === state.sport)
    .filter((prediction) => state.market === "all" || prediction.market === state.market)
    .filter((prediction) => state.book === "all" || prediction.book === state.book)
    .filter((prediction) => state.status === "all" || prediction.marketStatus === state.status)
    .filter((prediction) => (
      minimumProbability == null
      || (isFiniteNumber(prediction.modelProbability) && prediction.modelProbability >= minimumProbability)
    ))
    // The .filter() chain above already produced a fresh array, so sorting in place
    // is safe here and keeps the board working on browsers without Array#toSorted.
    .sort((left, right) => compareMarketRows(left, right, state.sort, state.dir));
}

export function paginateMarketRows(
  predictions: Prediction[],
  page: number,
  pageSize = MARKET_TABLE_PAGE_SIZE,
): Prediction[] {
  const safePage = Math.max(1, Math.floor(page));
  const start = (safePage - 1) * pageSize;
  return predictions.slice(start, start + pageSize);
}
