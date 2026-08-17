import type { JsonObject } from "@/lib/data/json";

export const MOBILE_SCHEMA_VERSION = "1.0";

export type MobileLeague = "NBA" | "NFL" | "MLB" | "PGA";
export type MobileFreshnessStatus = "fresh" | "stale" | "missing" | "offline";
export type MobileSource = "supabase" | "static_json" | "mixed" | "fixture" | "unavailable";

export type MobileFreshness = {
  status: MobileFreshnessStatus;
  source: MobileSource;
  updatedAt: string | null;
  ageSeconds: number | null;
};

export type MobileEnvelope<T> = {
  schemaVersion: typeof MOBILE_SCHEMA_VERSION;
  generatedAt: string;
  data: T;
  gaps: string[];
  freshness: MobileFreshness;
};

export type MobileMarket = {
  id: string;
  gameId: string;
  league: MobileLeague;
  kind: "team_spread" | "player_market";
  title: string;
  subtitle: string;
  eventTime: string | null;
  homeTeam: string | null;
  awayTeam: string | null;
  subject: string | null;
  market: string;
  book: string | null;
  line: number | null;
  price: number | null;
  modelProbability: number | null;
  impliedProbability: number | null;
  edge: number | null;
  ev: number | null;
  confidence: number | null;
  modelVersion: string | null;
  freshnessStatus: string;
  predictionTs: string | null;
  oddsTs: string | null;
  injuryAdjusted: boolean;
  injuryDataMissing: boolean;
};

export type MobileHomeData = {
  topEdges: MobileMarket[];
  leagueSummaries: {
    league: MobileLeague;
    marketCount: number;
    topEdge: number | null;
  }[];
};

export type MobileMarketsData = {
  league: MobileLeague;
  windowStart: string | null;
  windowEnd: string | null;
  markets: MobileMarket[];
};

export type MobileGameExplanation = {
  gameId: string;
  league: MobileLeague;
  modelVersion: string;
  predictionTs: string;
  topFeatures: {
    feature: string;
    value: number;
    impact: number;
    isHeuristic?: boolean;
  }[];
  injuryAdjusted: boolean;
  homeInjuryDelta: number | null;
  awayInjuryDelta: number | null;
  baseVsAdjusted: JsonObject | null;
};

export type MobileGameDetailData = {
  game: MobileMarket;
  explanation: MobileGameExplanation | null;
};

export type MobilePerformanceRecord = {
  league: string;
  modelVersion: string;
  season: string;
  market: string;
  sampleSize: number | null;
  roi: number | null;
  units: number | null;
  bets: number | null;
  wins: number | null;
  losses: number | null;
  pushes: number | null;
  productionStatus: string;
  gates: {
    id: string;
    label: string;
    status: string;
    detail: string;
  }[];
};

export type MobilePerformanceData = {
  generatedAt: string | null;
  records: MobilePerformanceRecord[];
};

export type MobileInsightsData = {
  dataQuality: {
    id: string;
    label: string;
    status: "ok" | "warning" | "blocked" | "missing";
    updatedAt: string | null;
    detail: string;
  }[];
  evaluations: {
    id: string;
    league: string;
    modelVersion: string;
    evaluationName: string;
    generatedAt: string;
    status: string;
    roi: number | null;
    auc: number | null;
  }[];
  strategies: {
    id: string;
    league: string;
    modelVersion: string;
    strategyId: string;
    market: string;
    sampleSize: number | null;
    bets: number | null;
    roi: number | null;
  }[];
};
