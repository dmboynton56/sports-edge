import type { MobileFreshness } from "@/lib/mobile/types";

export const MCP_MAX_EDGE_LIMIT = 25;
export const MCP_MAX_SLATE_ROWS = 40;
export const MCP_MAX_RESULT_ROWS = 40;
export const MCP_MAX_PLAYER_ROWS = 25;
export const MCP_MAX_FEATURES = 8;
export const MCP_MAX_LOOKAHEAD_DAYS = {
  NBA: 2,
  NFL: 7,
  MLB: 1,
  PGA: 7,
} as const;

export type McpLeague = "NBA" | "NFL" | "MLB" | "PGA";
export type McpEdgeSort = "point_edge" | "ev";
export type McpResultsWindow = "last_sunday" | "7d" | "season";
export type McpPublicationStatus = "research" | "model-only" | "candidate" | "production";
export type McpPlayerLeague = "MLB" | "NFL";
export type McpGameLeague = "NBA" | "NFL";

export type McpMeta = {
  as_of: string | null;
  model_version: string | null;
  freshness: MobileFreshness;
  status: McpPublicationStatus;
  gaps: string[];
};

export type McpEdgeRow = {
  id: string;
  league: string;
  title: string;
  homeTeam: string | null;
  awayTeam: string | null;
  market: string;
  line: number | null;
  modelProbability: number | null;
  pointEdge: number | null;
  ev: number | null;
  modelVersion: string | null;
  status: McpPublicationStatus;
};

export type McpSlateGame = {
  gameId: string;
  league: string;
  title: string;
  homeTeam: string | null;
  awayTeam: string | null;
  gameDate: string | null;
  eventTime: string | null;
  bookSpread: number | null;
  modelSpread: number | null;
  pointEdge: number | null;
  homeWinProb: number | null;
  modelVersion: string | null;
  freshnessStatus: string | null;
};

export type McpExplanationFeature = {
  feature: string;
  impact: number;
  value: number | null;
};

export type McpGameDetail = {
  gameId: string;
  league: string;
  title: string;
  homeTeam: string | null;
  awayTeam: string | null;
  line: number | null;
  modelSpread: number | null;
  homeWinProb: number | null;
  pointEdge: number | null;
  modelVersion: string | null;
  injuryAdjusted: boolean;
  injuryDataMissing: boolean;
  topFeatures: McpExplanationFeature[];
};

export type McpResultGame = {
  gameDate: string;
  homeTeam: string;
  awayTeam: string;
  homeScore: number;
  awayScore: number;
  bookSpread: number | null;
  modelSpread: number | null;
  spreadResult: "win" | "loss" | "push" | null;
  winnerResult: "win" | "loss" | null;
  units: number | null;
  modelVersion: string;
};

export type McpResultsSummary = {
  market: string;
  modelVersion: string;
  sample: number;
  wins: number;
  losses: number;
  pushes: number;
  units: number | null;
  hitRate: number | null;
  roi: number | null;
};

export type McpPlayerMarket = {
  id: string;
  league: string;
  player: string | null;
  team: string | null;
  opponent: string | null;
  market: string;
  modelProbability: number | null;
  fairOdds: string;
  modelVersion: string | null;
  asOf: string | null;
};

export type McpLeagueStatus = {
  league: McpLeague;
  as_of: string | null;
  model_version: string | null;
  freshness: MobileFreshness;
  status: McpPublicationStatus;
  last_publish: string | null;
  rowCount: number;
  gaps: string[];
};

export type McpPerformanceRecord = {
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
  productionStatus: string;
  gates: {
    id: string;
    label: string;
    status: string;
    detail: string;
  }[];
};

export type ListEdgesResult = McpMeta & {
  sort: McpEdgeSort;
  league: McpLeague | "ALL";
  limit: number;
  rows: McpEdgeRow[];
};

export type GetSlateResult = McpMeta & {
  league: McpLeague;
  windowStart: string | null;
  windowEnd: string | null;
  games: McpSlateGame[];
  event: {
    name: string | null;
    startDate: string | null;
    endDate: string | null;
  } | null;
};

export type GetGameResult = McpMeta & {
  game: McpGameDetail | null;
};

export type GetResultsResult = McpMeta & {
  league: string;
  window: McpResultsWindow;
  summaries: McpResultsSummary[];
  games: McpResultGame[];
};

export type GetStatusResult = McpMeta & {
  leagues: McpLeagueStatus[];
};

export type GetModelPerformanceResult = McpMeta & {
  records: McpPerformanceRecord[];
  backtests: {
    league: string;
    modelVersion: string;
    strategyId: string;
    market: string;
    sampleSize: number | null;
    bets: number | null;
    roi: number | null;
  }[];
  note: string;
};

export type GetPlayerMarketsResult = McpMeta & {
  league: McpPlayerLeague;
  rows: McpPlayerMarket[];
};

export type ListEdgesInput = {
  league?: McpLeague;
  sort: McpEdgeSort;
  limit: number;
};

export type GetSlateInput = {
  league: McpLeague;
  lookaheadDays?: number;
};

export type GetGameInput = {
  league: McpGameLeague | McpLeague;
  id?: string;
  team?: string;
};

export type GetResultsInput = {
  league: string;
  window: McpResultsWindow;
};

export type GetModelPerformanceInput = {
  league?: string;
};

export type GetPlayerMarketsInput = {
  league: McpPlayerLeague;
};
