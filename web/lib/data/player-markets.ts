import { promises as fs } from "fs";
import path from "path";

import type { Prediction } from "@/lib/data/types";
import {
  getMlbHomeRunModelLabel,
  MLB_HR_STATCAST_BLEND_MODEL,
  MLB_HR_V1_MODEL,
  type MlbHomeRunBoardData,
  type MlbHomeRunFeed,
  type MlbHomeRunModelFeed,
  type MlbHomeRunPrediction,
  type MlbHomeRunStatcastHealth,
  type MlbHrBoardSnapshot,
} from "@/lib/data/mlb-hr-board";
import { getSupabaseMissingEnv, getSupabaseRuntimeConfig } from "@/lib/data/supabase";

export {
  getMlbHomeRunModelLabel,
  MLB_HR_STATCAST_BLEND_MODEL,
  MLB_HR_V1_MODEL,
  type MlbHomeRunBoardData,
  type MlbHomeRunFeed,
  type MlbHomeRunModelFeed,
  type MlbHomeRunPrediction,
  type MlbHrBoardSnapshot,
};

const MLB_HR_PATH = path.join(process.cwd(), "public", "data", "mlb_home_runs.json");
const MLB_SLATE_TIME_ZONE = "America/Denver";
const PGA_TOURNAMENT_PATH = path.join(
  process.cwd(),
  "public",
  "data",
  "pga_tournaments",
  "current.json",
);
const PGA_SLATE_TIME_ZONE = "America/Denver";

async function supabaseRest<T>(resource: string): Promise<T[] | null> {
  const config = getSupabaseRuntimeConfig();
  if (!config.url || !config.anonKey) return null;
  const base = config.url.replace(/\/$/, "");
  try {
    const response = await fetch(`${base}/rest/v1/${resource}`, {
      headers: {
        apikey: config.anonKey,
        Authorization: `Bearer ${config.anonKey}`,
      },
      next: { revalidate: 60 },
    });
    if (!response.ok) return null;
    return (await response.json()) as T[];
  } catch {
    return null;
  }
}

function uniqueGaps(gaps: (string | null | undefined)[]): string[] {
  return Array.from(new Set(gaps.filter(Boolean) as string[]));
}

function supabaseConfigGaps(): string[] {
  const missingEnv = getSupabaseMissingEnv();
  return missingEnv.length
    ? [`Supabase live feed unavailable: missing ${missingEnv.join(", ")}.`]
    : [];
}

function todayInTimeZone(timeZone: string): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date());
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

export function getMlbHrSlateDate(now = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: MLB_SLATE_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(now);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

function predictionGameDate(row: Pick<MlbHomeRunPrediction, "gameDate">): string | null {
  if (row.gameDate) return row.gameDate.slice(0, 10);
  return null;
}

type SupabaseMlbHrRow = {
  game_id: string;
  game_date: string;
  event_time: string | null;
  player_id: string;
  player_name: string;
  team: string | null;
  opponent: string | null;
  venue: string | null;
  lineup_slot: number | null;
  lineup_status: string | null;
  opposing_probable_pitcher: string | null;
  hr_probability: number;
  baseline_probability: number | null;
  games_since_last_hr: number | null;
  last_hr_date: string | null;
  rank: number | null;
  v1_probability: number | null;
  v1_rank: number | null;
  statcast_probability: number | null;
  statcast_rank: number | null;
  statcast_available: boolean | null;
  model_agreement: string | null;
  consensus_score: number | null;
  market_signal_rank: number | null;
  confidence: number | null;
  model_version: string;
  prediction_ts: string | null;
  quality_flags: string[] | null;
  top_features: { feature: string; value: number }[] | null;
  statcast_coverage: number | null;
  statcast_ready_rows: number | null;
  statcast_total_rows: number | null;
  statcast_artifact_loaded: boolean | null;
};

type SupabaseMlbHrEdgeRow = SupabaseMlbHrRow & {
  best_book: string | null;
  best_book_title: string | null;
  best_price: number | null;
  implied_probability: number | null;
  no_vig_probability: number | null;
  market_probability: number | null;
  edge: number | null;
  ev: number | null;
  kelly: number | null;
  odds_books_count: number | null;
  odds_snapshot_ts: string | null;
  odds_status: string | null;
};

type SupabaseMlbHrBoardRunRow = {
  run_id: string;
  run_key: string;
  slate_date: string;
  model_version: string;
  run_window: "morning" | "afternoon" | "manual";
  status: "running" | "healthy" | "partial" | "failed" | "no_slate";
  started_at: string;
  completed_at: string | null;
  workflow_url: string | null;
  gaps: string[] | null;
  validation_summary: Record<string, unknown> | null;
  total_candidates: number;
  priced_candidates: number;
  top25_denominator: number;
  top25_priced_count: number;
  top25_coverage: number | null;
  prediction_ts: string | null;
  odds_ts: string | null;
};

type SupabaseMlbHrBoardRow = {
  board_row_id: string;
  run_id: string;
  run_key: string;
  run_slate_date: string;
  run_window: "morning" | "afternoon" | "manual";
  run_status: "healthy" | "partial";
  run_completed_at: string | null;
  run_prediction_ts: string | null;
  run_odds_ts: string | null;
  run_gaps: string[] | null;
  run_total_candidates: number;
  run_priced_candidates: number;
  run_top25_denominator: number;
  run_top25_priced_count: number;
  run_top25_coverage: number | null;
  slate_date: string;
  game_id: string;
  player_id: string;
  player_name: string;
  team: string | null;
  opponent: string | null;
  venue: string | null;
  event_time: string | null;
  lineup_slot: number | null;
  lineup_status: string | null;
  opposing_probable_pitcher: string | null;
  model_version: string;
  model_probability: number;
  baseline_probability: number | null;
  rank: number | null;
  book: string | null;
  american_price: number | null;
  raw_market_probability: number | null;
  no_vig_market_probability: number | null;
  market_probability: number | null;
  edge: number | null;
  ev: number | null;
  quarter_kelly: number | null;
  odds_snapshot_ts: string | null;
  odds_status: string;
  odds_books_count: number | null;
  quality_flags: string[] | null;
  statcast_available: boolean | null;
  statcast_coverage: number | null;
  prediction_ts: string | null;
  published_at: string;
};

type SupabasePgaTournamentRow = {
  event_key: string;
  season: number;
  name: string;
  start_date: string;
  end_date: string;
  course: string | null;
  par: number | null;
  field_size: number | null;
  status: string;
  raw_record: Record<string, unknown> | null;
  updated_at: string | null;
};

type SupabasePgaPredictionRow = {
  event_key: string;
  event_name: string;
  season: number;
  start_date: string;
  end_date: string;
  course: string | null;
  par: number | null;
  player_name: string;
  player_id: string | null;
  exp_sg_per_round: number | null;
  make_cut_prob: number | null;
  top5_prob: number | null;
  top10_prob: number | null;
  top20_prob: number | null;
  win_prob: number | null;
  projected_total_strokes: number | null;
  projected_score_to_par: number | null;
  model_version: string;
  prediction_ts: string | null;
  simulation_count: number | null;
  confidence: number | null;
  quality_flags: string[] | null;
};

export type PgaBoardData = Record<string, unknown> & {
  generatedAt: string | null;
  dataSource?: "supabase_predictions" | "static_json" | "unavailable";
  gaps?: string[];
  event?: Record<string, unknown>;
  predictions?: Record<string, unknown>[];
  predictionMeta?: Record<string, unknown>;
  normalizedMarkets?: Prediction[];
};

function mapSupabaseMlb(row: SupabaseMlbHrRow): MlbHomeRunPrediction {
  const isV1 = row.model_version.startsWith(MLB_HR_V1_MODEL);
  const isStatcast = row.model_version === MLB_HR_STATCAST_BLEND_MODEL;
  return {
    id: `${row.game_id}-${row.player_id}-${row.model_version}-hr`,
    sport: "MLB",
    league: "MLB",
    gameId: row.game_id,
    gameDate: row.game_date,
    eventTime: row.event_time,
    subject: `${row.player_name} HR`,
    player: row.player_name,
    market: "home_run",
    book: "model",
    line: 0.5,
    price: null,
    modelProbability: row.hr_probability,
    impliedProbability: null,
    edge: null,
    ev: null,
    kelly: null,
    confidence: row.confidence,
    modelVersion: row.model_version,
    source: "Supabase mlb_home_run_predictions_latest",
    updatedAt: row.prediction_ts,
    team: row.team,
    opponent: row.opponent,
    venue: row.venue,
    lineupSlot: row.lineup_slot,
    lineupStatus: row.lineup_status,
    opposingProbablePitcher: row.opposing_probable_pitcher,
    baselineProbability: row.baseline_probability,
    gamesSinceLastHr: row.games_since_last_hr,
    lastHrDate: row.last_hr_date,
    rank: row.rank,
    qualityFlags: row.quality_flags ?? [],
    topFeatures: row.top_features ?? [],
    v1Probability: row.v1_probability ?? (isV1 ? row.hr_probability : null),
    v1Rank: row.v1_rank ?? (isV1 ? row.rank : null),
    statcastProbability: row.statcast_probability ?? (isStatcast ? row.hr_probability : null),
    statcastRank: row.statcast_rank ?? (isStatcast ? row.rank : null),
    statcastAvailable: row.statcast_available,
    modelAgreement: row.model_agreement ?? (isV1 ? "V1 only" : null),
    consensusScore: row.consensus_score ?? row.rank,
    marketSignalRank: row.market_signal_rank,
    statcastCoverage: row.statcast_coverage,
    statcastReadyRows: row.statcast_ready_rows,
    statcastTotalRows: row.statcast_total_rows,
    statcastArtifactLoaded: row.statcast_artifact_loaded,
  };
}

function mapSupabaseMlbEdge(row: SupabaseMlbHrEdgeRow): MlbHomeRunPrediction {
  const base = mapSupabaseMlb(row);
  return {
    ...base,
    book: row.best_book ?? "missing",
    price: row.best_price,
    impliedProbability: row.market_probability ?? row.implied_probability,
    edge: row.edge,
    ev: row.ev,
    kelly: row.kelly,
    source: "Supabase mlb_home_run_edges_latest",
    bestBook: row.best_book,
    bestBookTitle: row.best_book_title,
    bestPrice: row.best_price,
    noVigProbability: row.no_vig_probability,
    marketProbability: row.market_probability,
    oddsBooksCount: row.odds_books_count,
    oddsSnapshotTs: row.odds_snapshot_ts,
    oddsStatus: row.odds_status,
  };
}

function stringList(value: string[] | string | null | undefined): string[] {
  if (Array.isArray(value)) return value.filter(Boolean);
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.filter(Boolean) : [value];
  } catch {
    return [value];
  }
}

function mapBoardRow(row: SupabaseMlbHrBoardRow): MlbHomeRunPrediction {
  const isV1 = row.model_version.startsWith(MLB_HR_V1_MODEL);
  const priced = row.odds_status === "ok" || row.odds_status === "raw_implied";
  return {
    id: row.board_row_id,
    sport: "MLB",
    league: "MLB",
    gameId: row.game_id,
    gameDate: row.slate_date,
    eventTime: row.event_time,
    subject: `${row.player_name} HR`,
    player: row.player_name,
    market: "home_run",
    book: row.book ?? "model",
    line: 0.5,
    price: priced ? row.american_price : null,
    modelProbability: row.model_probability,
    impliedProbability: priced ? row.market_probability ?? row.raw_market_probability : null,
    edge: priced ? row.edge : null,
    ev: priced ? row.ev : null,
    kelly: priced ? row.quarter_kelly : null,
    confidence: null,
    modelVersion: row.model_version,
    source: "Supabase mlb_home_run_board_latest",
    updatedAt: row.prediction_ts,
    team: row.team,
    opponent: row.opponent,
    venue: row.venue,
    lineupSlot: row.lineup_slot,
    lineupStatus: row.lineup_status,
    opposingProbablePitcher: row.opposing_probable_pitcher,
    baselineProbability: row.baseline_probability,
    rank: row.rank,
    qualityFlags: stringList(row.quality_flags),
    topFeatures: [],
    v1Probability: isV1 ? row.model_probability : null,
    v1Rank: isV1 ? row.rank : null,
    statcastProbability: null,
    statcastRank: null,
    statcastAvailable: row.statcast_available,
    modelAgreement: isV1 ? "V1 only" : null,
    consensusScore: row.rank,
    marketSignalRank: null,
    statcastCoverage: row.statcast_coverage,
    statcastReadyRows: null,
    statcastTotalRows: null,
    statcastArtifactLoaded: null,
    bestBook: priced ? row.book : null,
    bestPrice: priced ? row.american_price : null,
    noVigProbability: priced ? row.no_vig_market_probability : null,
    marketProbability: priced ? row.market_probability : null,
    oddsBooksCount: priced ? row.odds_books_count : null,
    oddsSnapshotTs: priced ? row.odds_snapshot_ts : null,
    oddsStatus: row.odds_status,
  };
}

function localClock(now: Date) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: MLB_SLATE_TIME_ZONE,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(now);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  const hour = Number(values.hour);
  return { hour: hour === 24 ? 0 : hour, minute: Number(values.minute) };
}

function localMinutes(value: string | null | undefined): number | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  const clock = localClock(date);
  return clock.hour * 60 + clock.minute;
}

function scheduledRunIsEligible(run: SupabaseMlbHrBoardRunRow, now: Date): boolean {
  const slateDate = getMlbHrSlateDate(now);
  if (run.slate_date !== slateDate || !run.completed_at) return false;
  const clock = localClock(now);
  const minutes = localMinutes(run.completed_at);
  if (minutes == null) return false;
  if (clock.hour < 8) return true;
  if (clock.hour < 16) return (run.run_window === "morning" || run.run_window === "manual") && minutes >= 360;
  return (run.run_window === "afternoon" || run.run_window === "manual") && minutes >= 840;
}

function boardStatusForRun(
  run: SupabaseMlbHrBoardRunRow | null,
  now: Date,
): { status: MlbHrBoardSnapshot["status"]; gaps: string[] } {
  const slateDate = getMlbHrSlateDate(now);
  const beforeBoardOpen = localClock(now).hour < 8;
  if (!run) {
    return {
      status: beforeBoardOpen ? "stale" : "unavailable",
      gaps: [beforeBoardOpen ? "Board updating before 8:00 AM Mountain." : `No completed MLB HR run for ${slateDate}.`],
    };
  }
  const gaps = stringList(run.gaps);
  if (run.status === "no_slate") return { status: "no_slate", gaps };
  if (run.status === "running" || run.status === "failed") {
    return { status: "stale", gaps: uniqueGaps([...gaps, "The latest MLB HR run did not complete successfully."]) };
  }
  if (!scheduledRunIsEligible(run, now)) {
    return {
      status: "stale",
      gaps: uniqueGaps([
        ...gaps,
        localClock(now).hour < 16
          ? "Waiting for a current morning/manual run completed after 6:00 AM Mountain."
          : "Waiting for a current afternoon/manual run completed after 2:00 PM Mountain.",
      ]),
    };
  }
  return { status: run.status === "partial" ? "partial" : "healthy", gaps };
}

export function deriveMlbHrBoardSnapshot(
  run: SupabaseMlbHrBoardRunRow | null,
  sourceRows: SupabaseMlbHrBoardRow[],
  now = new Date(),
): MlbHrBoardSnapshot {
  const slateDate = getMlbHrSlateDate(now);
  const boardStatus = boardStatusForRun(run, now);
  if (!run || boardStatus.status === "stale" || boardStatus.status === "unavailable") {
    return {
      slateDate,
      status: boardStatus.status,
      modelStatus: "candidate",
      runWindow: run?.run_window ?? "manual",
      predictionAsOf: run?.prediction_ts ?? null,
      oddsAsOf: run?.odds_ts ?? null,
      counts: {
        candidates: 0,
        priced: 0,
        top25Eligible: run?.top25_denominator ?? 0,
        top25Priced: run?.top25_priced_count ?? 0,
        top25Coverage: run?.top25_coverage ?? null,
      },
      rows: [],
      gaps: boardStatus.gaps,
      dataSource: run ? "supabase_board" : "unavailable",
      completedAt: run?.completed_at ?? null,
    };
  }

  const cutoff = now.getTime() + 5 * 60 * 1000;
  const rows = sourceRows
    .filter((row) => row.model_version.startsWith(MLB_HR_V1_MODEL))
    .filter((row) => {
      const eventTime = row.event_time ? new Date(row.event_time).getTime() : Number.NaN;
      return Number.isFinite(eventTime) && eventTime > cutoff;
    })
    .sort((left, right) => (left.rank ?? Number.MAX_SAFE_INTEGER) - (right.rank ?? Number.MAX_SAFE_INTEGER));
  const predictions = rows.map(mapBoardRow);
  const priced = predictions.filter((row) => row.oddsStatus === "ok" || row.oddsStatus === "raw_implied");
  return {
    slateDate,
    status: boardStatus.status,
    modelStatus: "candidate",
    runWindow: run.run_window,
    predictionAsOf: run.prediction_ts,
    oddsAsOf: run.odds_ts,
    counts: {
      candidates: predictions.length,
      priced: priced.length,
      top25Eligible: run.top25_denominator,
      top25Priced: run.top25_priced_count,
      top25Coverage: run.top25_coverage,
    },
    rows: predictions,
    gaps: uniqueGaps([
      ...boardStatus.gaps,
      predictions.length < sourceRows.filter((row) => row.model_version.startsWith(MLB_HR_V1_MODEL)).length
        ? "Rows for games that have started or begin within five minutes are hidden."
        : null,
    ]),
    dataSource: "supabase_board",
    completedAt: run.completed_at,
  };
}

export async function getMlbHomeRunBoardSnapshot(now = new Date()): Promise<MlbHrBoardSnapshot> {
  const slateDate = getMlbHrSlateDate(now);
  if (process.env.MLB_HR_TRUSTED_BOARD_ENABLED === "false") {
    return {
      slateDate,
      status: "unavailable",
      modelStatus: "candidate",
      runWindow: "manual",
      predictionAsOf: null,
      oddsAsOf: null,
      counts: { candidates: 0, priced: 0, top25Eligible: 0, top25Priced: 0, top25Coverage: null },
      rows: [],
      gaps: ["Trusted MLB HR board is disabled by MLB_HR_TRUSTED_BOARD_ENABLED."],
      dataSource: "unavailable",
      completedAt: null,
    };
  }

  const runs = await supabaseRest<SupabaseMlbHrBoardRunRow>(
    `mlb_home_run_board_run_health?select=*&slate_date=eq.${slateDate}&limit=1`,
  );
  if (!runs) {
    const unavailable = deriveMlbHrBoardSnapshot(null, [], now);
    return { ...unavailable, gaps: uniqueGaps([...unavailable.gaps, ...supabaseConfigGaps()]) };
  }
  const run = runs[0] ?? null;
  if (!run) {
    const unavailable = deriveMlbHrBoardSnapshot(null, [], now);
    return { ...unavailable, gaps: uniqueGaps([...unavailable.gaps, ...supabaseConfigGaps()]) };
  }
  const rows = await supabaseRest<SupabaseMlbHrBoardRow>(
    `mlb_home_run_board_latest?select=*&run_slate_date=eq.${slateDate}&order=rank.asc&limit=500`,
  );
  if (!rows) {
    return {
      ...deriveMlbHrBoardSnapshot(run, [], now),
      status: "unavailable",
      gaps: uniqueGaps([...(stringList(run.gaps)), "Published MLB HR board rows are unavailable."]),
      dataSource: "unavailable",
    };
  }
  return deriveMlbHrBoardSnapshot(run, rows, now);
}

function buildFeedFromPayload(
  payload: MlbHomeRunFeed,
  slateDate: string,
  modelVersion?: string,
  fallbackGaps: string[] = [],
): MlbHomeRunFeed {
  const defaultModel = payload.defaultModel ?? MLB_HR_V1_MODEL;
  const targetModel = modelVersion ?? defaultModel;

  if (payload.models?.[targetModel]) {
    const modelPayload = payload.models[targetModel];
    const predictions = (modelPayload.predictions ?? []).filter(
      (row) => predictionGameDate(row) === slateDate,
    );
    return {
      generatedAt: payload.generatedAt,
      defaultModel,
      modelVersion: modelPayload.modelVersion ?? targetModel,
      productionStatus: payload.productionStatus ?? "candidate",
      predictions,
      gaps: uniqueGaps([...fallbackGaps, ...(modelPayload.gaps ?? [])]),
      dataSource: "static_json",
      models: payload.models,
      statcastHealth: payload.statcastHealth,
    };
  }

  const predictions = (payload.predictions ?? []).filter(
    (row) => predictionGameDate(row) === slateDate,
  );
  const existingGaps = payload.gaps ?? [];
  return {
    ...payload,
    defaultModel,
    predictions,
    gaps: uniqueGaps([
      ...fallbackGaps,
      ...existingGaps,
      predictions.length ? null : `No MLB home run predictions available for ${slateDate}.`,
    ]),
    dataSource: "static_json",
    statcastHealth: payload.statcastHealth,
  };
}

function buildBoardFromPayload(
  payload: MlbHomeRunFeed,
  slateDate: string,
  fallbackGaps: string[] = [],
): MlbHomeRunBoardData {
  const defaultModel = payload.defaultModel ?? MLB_HR_V1_MODEL;
  const models: Record<string, MlbHomeRunModelFeed> = {};

  const defaultPredictions = (payload.predictions ?? []).filter(
    (row) => predictionGameDate(row) === slateDate,
  );
  if (defaultPredictions.length) {
    return {
      generatedAt: payload.generatedAt,
      productionStatus: payload.productionStatus ?? "candidate",
      defaultModel,
      availableModels: [defaultModel],
      models: {
        [defaultModel]: {
          modelVersion: payload.modelVersion ?? defaultModel,
          predictions: defaultPredictions,
          gaps: payload.gaps ?? [],
        },
      },
      gaps: fallbackGaps,
      dataSource: "static_json",
      statcastHealth: payload.statcastHealth,
    };
  }

  if (payload.models && Object.keys(payload.models).length) {
    for (const [modelKey, modelPayload] of Object.entries(payload.models)) {
      const predictions = (modelPayload.predictions ?? []).filter(
        (row) => predictionGameDate(row) === slateDate,
      );
      if (!predictions.length) continue;
      models[modelKey] = {
        modelVersion: modelPayload.modelVersion ?? modelKey,
        predictions,
        gaps: modelPayload.gaps ?? [],
      };
    }
  }

  if (!Object.keys(models).length) {
    return {
      generatedAt: payload.generatedAt,
      productionStatus: payload.productionStatus ?? "candidate",
      defaultModel,
      availableModels: [],
      models: {},
      gaps: uniqueGaps([
        ...fallbackGaps,
        `No MLB home run predictions available for ${slateDate}.`,
      ]),
      dataSource: "static_json",
      statcastHealth: payload.statcastHealth,
    };
  }

  const availableModels = Object.keys(models);
  return {
    generatedAt: payload.generatedAt,
    productionStatus: payload.productionStatus ?? "candidate",
    defaultModel: availableModels.includes(defaultModel) ? defaultModel : availableModels[0] ?? defaultModel,
    availableModels,
    models,
    gaps: fallbackGaps,
    dataSource: "static_json",
    statcastHealth: payload.statcastHealth,
  };
}

function healthFromRows(rows: MlbHomeRunPrediction[]): MlbHomeRunStatcastHealth | undefined {
  const row = rows.find((candidate) => candidate.statcastTotalRows != null);
  if (!row) return undefined;
  return {
    enabled: true,
    artifactLoaded: Boolean(row.statcastArtifactLoaded),
    coverage: row.statcastCoverage ?? null,
    readyRows: row.statcastReadyRows ?? 0,
    totalRows: row.statcastTotalRows ?? rows.length,
  };
}

function buildBoardFromSupabaseRows(
  rows: MlbHomeRunPrediction[],
  generatedAt: string | null,
  dataSource: MlbHomeRunBoardData["dataSource"],
): MlbHomeRunBoardData {
  const models: Record<string, MlbHomeRunModelFeed> = {};

  for (const row of rows) {
    const modelKey = row.modelVersion ?? MLB_HR_V1_MODEL;
    const model = models[modelKey] ?? {
      modelVersion: modelKey,
      predictions: [],
      gaps: [],
    };
    model.predictions.push(row);
    models[modelKey] = model;
  }

  for (const model of Object.values(models)) {
    const missingOdds = model.predictions.filter(
      (row) => row.oddsStatus === "missing_odds",
    ).length;
    const missingStatcast = model.predictions.filter(
      (row) => row.modelAgreement === "Missing Statcast" || row.statcastAvailable === false,
    ).length;
    model.gaps = [
      missingOdds
        ? `Missing sportsbook odds for ${missingOdds} MLB home run candidates.`
        : null,
      missingStatcast
        ? `Statcast features unavailable for ${missingStatcast} candidates; those rows use the V1 fallback.`
        : null,
    ].filter(Boolean) as string[];
  }

  const availableModels = Object.keys(models);
  const defaultModel = availableModels.includes(MLB_HR_V1_MODEL)
    ? MLB_HR_V1_MODEL
    : availableModels[0] ?? MLB_HR_V1_MODEL;

  return {
    generatedAt,
    productionStatus: "candidate",
    defaultModel,
    availableModels,
    models,
    gaps: [],
    dataSource,
    statcastHealth: healthFromRows(rows),
  };
}

function modelVersionFilter(modelVersion?: string): string {
  return modelVersion ? `&model_version=eq.${encodeURIComponent(modelVersion)}` : "";
}

export async function getMlbHomeRunFeed(modelVersion?: string): Promise<MlbHomeRunFeed> {
  const slateDate = todayInTimeZone(MLB_SLATE_TIME_ZONE);
  // The mobile feed is a single-model surface. Without an explicit filter,
  // the serving views contain both the V1 and Statcast-blend rows, and the
  // rank-limited query can return duplicate players from the two models.
  const targetModel = modelVersion ?? MLB_HR_V1_MODEL;
  const versionQuery = modelVersionFilter(targetModel);
  const edgeRows = await supabaseRest<SupabaseMlbHrEdgeRow>(
    `mlb_home_run_edges_latest?select=*&game_date=eq.${slateDate}${versionQuery}&order=rank.asc&limit=120`,
  );
  if (edgeRows && edgeRows.length) {
    const missingOdds = edgeRows.filter((row) => row.odds_status === "missing_odds").length;
    return {
      generatedAt: edgeRows[0]?.prediction_ts ?? null,
      defaultModel: targetModel,
      modelVersion: edgeRows[0]?.model_version ?? "mlb-hr-v1-heuristic",
      productionStatus: "candidate",
      predictions: edgeRows.map(mapSupabaseMlbEdge),
      gaps: missingOdds
        ? [`Missing sportsbook odds for ${missingOdds} MLB home run candidates.`]
        : [],
      dataSource: "supabase_edges",
      statcastHealth: healthFromRows(edgeRows.map(mapSupabaseMlbEdge)),
    };
  }

  const latestRows = await supabaseRest<SupabaseMlbHrRow>(
    `mlb_home_run_predictions_latest?select=*&game_date=eq.${slateDate}${versionQuery}&order=rank.asc&limit=120`,
  );
  const rows = latestRows?.length
    ? latestRows
    : await supabaseRest<SupabaseMlbHrRow>(
        `mlb_home_run_predictions?select=*&game_date=eq.${slateDate}${versionQuery}&order=rank.asc&limit=120`,
      );
  if (rows && rows.length) {
    return {
      generatedAt: rows[0]?.prediction_ts ?? null,
      defaultModel: targetModel,
      modelVersion: rows[0]?.model_version ?? "mlb-hr-v1-heuristic",
      productionStatus: "candidate",
      predictions: rows.map(mapSupabaseMlb),
      gaps: [],
      dataSource: "supabase_predictions",
      statcastHealth: healthFromRows(rows.map(mapSupabaseMlb)),
    };
  }

  if (process.env.MLB_HR_USE_LOCAL_FIXTURE !== "true") {
    return {
      generatedAt: null,
      defaultModel: modelVersion ?? MLB_HR_V1_MODEL,
      modelVersion: "mlb-hr-v1-heuristic",
      productionStatus: "candidate",
      predictions: [],
      gaps: uniqueGaps([
        ...supabaseConfigGaps(),
        "Supabase MLB HR source returned no current rows; local JSON is disabled for production serving.",
      ]),
      dataSource: "unavailable",
    };
  }

  try {
    const payload = JSON.parse(await fs.readFile(MLB_HR_PATH, "utf8")) as MlbHomeRunFeed;
    return buildFeedFromPayload(payload, slateDate, targetModel, supabaseConfigGaps());
  } catch {
    return {
      generatedAt: null,
      defaultModel: targetModel,
      modelVersion: "mlb-hr-v1-heuristic",
      productionStatus: "candidate",
      predictions: [],
      gaps: uniqueGaps([
        ...supabaseConfigGaps(),
        "No MLB home run artifact found at web/public/data/mlb_home_runs.json.",
      ]),
      dataSource: "unavailable",
    };
  }
}

export async function getMlbHomeRunBoardData(): Promise<MlbHomeRunBoardData> {
  const slateDate = todayInTimeZone(MLB_SLATE_TIME_ZONE);
  const edgeRows = await supabaseRest<SupabaseMlbHrEdgeRow>(
    `mlb_home_run_edges_latest?select=*&game_date=eq.${slateDate}&order=model_version.asc,rank.asc&limit=300`,
  );
  if (edgeRows && edgeRows.length) {
    return buildBoardFromSupabaseRows(
      edgeRows.map(mapSupabaseMlbEdge),
      edgeRows[0]?.prediction_ts ?? null,
      "supabase_edges",
    );
  }

  const latestRows = await supabaseRest<SupabaseMlbHrRow>(
    `mlb_home_run_predictions_latest?select=*&game_date=eq.${slateDate}&order=model_version.asc,rank.asc&limit=300`,
  );
  const rows = latestRows?.length
    ? latestRows
    : await supabaseRest<SupabaseMlbHrRow>(
        `mlb_home_run_predictions?select=*&game_date=eq.${slateDate}&order=model_version.asc,rank.asc&limit=300`,
      );
  if (rows && rows.length) {
    return buildBoardFromSupabaseRows(
      rows.map(mapSupabaseMlb),
      rows[0]?.prediction_ts ?? null,
      "supabase_predictions",
    );
  }

  if (process.env.MLB_HR_USE_LOCAL_FIXTURE !== "true") {
    return {
      generatedAt: null,
      productionStatus: "candidate",
      defaultModel: MLB_HR_V1_MODEL,
      availableModels: [],
      models: {},
      gaps: [
        ...supabaseConfigGaps(),
        "Supabase MLB HR source returned no current rows; local JSON is disabled for production serving.",
      ],
      dataSource: "unavailable",
    };
  }

  try {
    const payload = JSON.parse(await fs.readFile(MLB_HR_PATH, "utf8")) as MlbHomeRunFeed;
    return buildBoardFromPayload(payload, slateDate, supabaseConfigGaps());
  } catch {
    const fallback = await getMlbHomeRunFeed();
    return {
      generatedAt: fallback.generatedAt,
      productionStatus: fallback.productionStatus,
      defaultModel: MLB_HR_V1_MODEL,
      availableModels: fallback.predictions.length ? [MLB_HR_V1_MODEL] : [],
      models: fallback.predictions.length
        ? {
            [MLB_HR_V1_MODEL]: {
              modelVersion: fallback.modelVersion,
              predictions: fallback.predictions,
              gaps: fallback.gaps,
            },
          }
        : {},
      gaps: fallback.gaps,
      dataSource: fallback.dataSource ?? "unavailable",
      statcastHealth: fallback.statcastHealth,
    };
  }
}

export async function getPgaNormalizedMarkets(): Promise<Prediction[]> {
  try {
    const payload = JSON.parse(await fs.readFile(PGA_TOURNAMENT_PATH, "utf8")) as {
      normalizedMarkets?: Prediction[];
    };
    return payload.normalizedMarkets ?? [];
  } catch {
    return [];
  }
}

async function readPgaStaticPayload(fallbackGaps: string[] = []): Promise<PgaBoardData> {
  try {
    const payload = JSON.parse(await fs.readFile(PGA_TOURNAMENT_PATH, "utf8")) as PgaBoardData;
    return {
      ...payload,
      dataSource: "static_json",
      gaps: uniqueGaps([...(payload.gaps ?? []), ...fallbackGaps]),
    };
  } catch {
    return {
      generatedAt: null,
      dataSource: "unavailable",
      predictions: [],
      normalizedMarkets: [],
      gaps: uniqueGaps([
        ...fallbackGaps,
        "No PGA artifact found at web/public/data/pga_tournaments/current.json.",
      ]),
    };
  }
}

function mapSupabasePgaPrediction(row: SupabasePgaPredictionRow): Record<string, unknown> {
  return {
    player: row.player_name,
    player_id: row.player_id,
    exp_sg_per_round: row.exp_sg_per_round,
    sim_win_pct: (row.win_prob ?? 0) * 100,
    sim_top5_pct: (row.top5_prob ?? 0) * 100,
    sim_top10_pct: (row.top10_prob ?? 0) * 100,
    sim_top20_pct: (row.top20_prob ?? 0) * 100,
    projected_total_strokes: row.projected_total_strokes,
    projected_score_to_par: row.projected_score_to_par,
    confidence: row.confidence,
    quality_flags: row.quality_flags ?? [],
    source: "Supabase pga_player_predictions_latest",
    best_calibrated_target_made_cut_prob: row.make_cut_prob,
    best_calibrated_target_top10_prob: row.top10_prob,
    best_calibrated_target_top20_prob: row.top20_prob,
    best_calibrated_target_win_prob: row.win_prob,
  };
}

function mapSupabasePgaMarket(row: SupabasePgaPredictionRow, market: string, probability: number | null): Prediction | null {
  if (probability == null) return null;
  return {
    id: `PGA-${row.event_key}-${row.player_id ?? row.player_name}-${market}`,
    sport: "PGA",
    league: "PGA",
    gameId: row.event_key,
    eventTime: row.start_date,
    subject: row.player_name,
    player: row.player_name,
    market,
    book: "model",
    line: null,
    price: null,
    modelProbability: probability,
    impliedProbability: null,
    edge: null,
    ev: null,
    kelly: null,
    confidence: row.confidence,
    modelVersion: row.model_version,
    source: "Supabase pga_player_predictions_latest",
    updatedAt: row.prediction_ts,
  };
}

async function getCurrentPgaTournament(): Promise<SupabasePgaTournamentRow | null> {
  const today = todayInTimeZone(PGA_SLATE_TIME_ZONE);
  const activeRows = await supabaseRest<SupabasePgaTournamentRow>(
    `pga_tournaments?select=*&start_date=lte.${today}&end_date=gte.${today}&order=updated_at.desc&limit=1`,
  );
  if (activeRows?.length) return activeRows[0];
  const latestRows = await supabaseRest<SupabasePgaTournamentRow>(
    "pga_tournaments?select=*&order=start_date.desc&limit=1",
  );
  return latestRows?.[0] ?? null;
}

export async function getPgaBoardData(): Promise<PgaBoardData> {
  const staticPayload = await readPgaStaticPayload(supabaseConfigGaps());
  const tournament = await getCurrentPgaTournament();
  if (!tournament) {
    return staticPayload;
  }

  const latestRows = await supabaseRest<SupabasePgaPredictionRow>(
    `pga_player_predictions_latest?select=*&event_key=eq.${encodeURIComponent(tournament.event_key)}&order=win_prob.desc.nullslast&limit=250`,
  );
  const rows = latestRows?.length
    ? latestRows
    : await supabaseRest<SupabasePgaPredictionRow>(
        `pga_player_predictions?select=*&event_key=eq.${encodeURIComponent(tournament.event_key)}&order=win_prob.desc.nullslast&limit=250`,
      );
  if (!rows?.length) {
    return staticPayload;
  }

  const generatedAt = rows[0]?.prediction_ts ?? tournament.updated_at ?? staticPayload.generatedAt;
  const normalizedMarkets = rows.flatMap((row) => [
    mapSupabasePgaMarket(row, "win", row.win_prob),
    mapSupabasePgaMarket(row, "top10", row.top10_prob),
    mapSupabasePgaMarket(row, "top20", row.top20_prob),
    mapSupabasePgaMarket(row, "make_cut", row.make_cut_prob),
  ]).filter(Boolean) as Prediction[];

  return {
    ...staticPayload,
    generatedAt,
    dataSource: "supabase_predictions",
    event: {
      ...(staticPayload.event ?? {}),
      eventKey: tournament.event_key,
      name: tournament.name,
      season: tournament.season,
      course: tournament.course ?? "",
      par: tournament.par,
      startDate: tournament.start_date,
      endDate: tournament.end_date,
      status: tournament.status,
    },
    predictions: rows.map(mapSupabasePgaPrediction),
    normalizedMarkets,
    predictionMeta: {
      ...(staticPayload.predictionMeta ?? {}),
      model_version: rows[0]?.model_version,
      n_players: rows.length,
      prediction_ts: generatedAt,
      source: "supabase",
    },
    gaps: uniqueGaps([...(staticPayload.gaps ?? [])]),
  };
}

export async function getProductionPredictionFeed(): Promise<{
  generatedAt: string | null;
  predictions: Prediction[];
  gaps: string[];
}> {
  const [mlb, pgaBoard] = await Promise.all([getMlbHomeRunFeed(), getPgaBoardData()]);
  const pga = pgaBoard.dataSource === "supabase_predictions" ? pgaBoard.normalizedMarkets ?? [] : [];
  const mlbPredictions = mlb.dataSource === "supabase_edges" || mlb.dataSource === "supabase_predictions"
    ? mlb.predictions
    : [];
  return {
    generatedAt: mlb.generatedAt,
    predictions: [...mlbPredictions, ...pga],
    gaps: [
      ...mlb.gaps,
      ...(mlbPredictions.length ? [] : ["No current Supabase MLB HR rows; static artifacts are not a production fallback."]),
      ...(pgaBoard.gaps ?? []),
    ],
  };
}
