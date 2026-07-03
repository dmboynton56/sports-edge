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
  const response = await fetch(`${base}/rest/v1/${resource}`, {
    headers: {
      apikey: config.anonKey,
      Authorization: `Bearer ${config.anonKey}`,
    },
    next: { revalidate: 60 },
  });
  if (!response.ok) return null;
  return (await response.json()) as T[];
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
    id: `${row.game_id}-${row.player_id}-hr`,
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
  const versionQuery = modelVersionFilter(modelVersion);
  const edgeRows = await supabaseRest<SupabaseMlbHrEdgeRow>(
    `mlb_home_run_edges_latest?select=*&game_date=eq.${slateDate}${versionQuery}&order=rank.asc&limit=120`,
  );
  if (edgeRows && edgeRows.length) {
    const missingOdds = edgeRows.filter((row) => row.odds_status === "missing_odds").length;
    return {
      generatedAt: edgeRows[0]?.prediction_ts ?? null,
      defaultModel: modelVersion ?? MLB_HR_V1_MODEL,
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

  const rows = await supabaseRest<SupabaseMlbHrRow>(
    `mlb_home_run_predictions_latest?select=*&game_date=eq.${slateDate}${versionQuery}&order=rank.asc&limit=120`,
  );
  if (rows && rows.length) {
    return {
      generatedAt: rows[0]?.prediction_ts ?? null,
      defaultModel: modelVersion ?? MLB_HR_V1_MODEL,
      modelVersion: rows[0]?.model_version ?? "mlb-hr-v1-heuristic",
      productionStatus: "candidate",
      predictions: rows.map(mapSupabaseMlb),
      gaps: [],
      dataSource: "supabase_predictions",
      statcastHealth: healthFromRows(rows.map(mapSupabaseMlb)),
    };
  }

  try {
    const payload = JSON.parse(await fs.readFile(MLB_HR_PATH, "utf8")) as MlbHomeRunFeed;
    return buildFeedFromPayload(payload, slateDate, modelVersion, supabaseConfigGaps());
  } catch {
    return {
      generatedAt: null,
      defaultModel: modelVersion ?? MLB_HR_V1_MODEL,
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

  const rows = await supabaseRest<SupabaseMlbHrRow>(
    `mlb_home_run_predictions_latest?select=*&game_date=eq.${slateDate}&order=model_version.asc,rank.asc&limit=300`,
  );
  if (rows && rows.length) {
    return buildBoardFromSupabaseRows(
      rows.map(mapSupabaseMlb),
      rows[0]?.prediction_ts ?? null,
      "supabase_predictions",
    );
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

  const rows = await supabaseRest<SupabasePgaPredictionRow>(
    `pga_player_predictions_latest?select=*&event_key=eq.${encodeURIComponent(tournament.event_key)}&order=win_prob.desc.nullslast&limit=250`,
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
  const pga = pgaBoard.normalizedMarkets ?? [];
  return {
    generatedAt: mlb.generatedAt,
    predictions: [...mlb.predictions, ...pga],
    gaps: [...mlb.gaps, ...(pgaBoard.gaps ?? [])],
  };
}
