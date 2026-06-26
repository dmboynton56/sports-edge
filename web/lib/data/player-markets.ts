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
} from "@/lib/data/mlb-hr-board";
import { getSupabaseRuntimeConfig } from "@/lib/data/supabase";

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
  confidence: number | null;
  model_version: string;
  prediction_ts: string | null;
  quality_flags: string[] | null;
  top_features: { feature: string; value: number }[] | null;
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

function mapSupabaseMlb(row: SupabaseMlbHrRow): MlbHomeRunPrediction {
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
    v1Probability: row.hr_probability,
    v1Rank: row.rank,
    statcastProbability: null,
    statcastRank: null,
    statcastAvailable: null,
    modelAgreement: "V1 only",
    consensusScore: row.rank,
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
      gaps: modelPayload.gaps ?? [],
      models: payload.models,
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
    gaps: predictions.length
      ? existingGaps
      : [...existingGaps, `No MLB home run predictions available for ${slateDate}.`],
  };
}

function buildBoardFromPayload(payload: MlbHomeRunFeed, slateDate: string): MlbHomeRunBoardData {
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
    };
  }

  const availableModels = Object.keys(models);
  return {
    generatedAt: payload.generatedAt,
    productionStatus: payload.productionStatus ?? "candidate",
    defaultModel: availableModels.includes(defaultModel) ? defaultModel : availableModels[0] ?? defaultModel,
    availableModels,
    models,
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
    };
  }

  try {
    const payload = JSON.parse(await fs.readFile(MLB_HR_PATH, "utf8")) as MlbHomeRunFeed;
    return buildFeedFromPayload(payload, slateDate, modelVersion);
  } catch {
    return {
      generatedAt: null,
      defaultModel: modelVersion ?? MLB_HR_V1_MODEL,
      modelVersion: "mlb-hr-v1-heuristic",
      productionStatus: "candidate",
      predictions: [],
      gaps: ["No MLB home run artifact found at web/public/data/mlb_home_runs.json."],
    };
  }
}

export async function getMlbHomeRunBoardData(): Promise<MlbHomeRunBoardData> {
  const slateDate = todayInTimeZone(MLB_SLATE_TIME_ZONE);
  try {
    const payload = JSON.parse(await fs.readFile(MLB_HR_PATH, "utf8")) as MlbHomeRunFeed;
    return buildBoardFromPayload(payload, slateDate);
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

export async function getProductionPredictionFeed(): Promise<{
  generatedAt: string | null;
  predictions: Prediction[];
  gaps: string[];
}> {
  const [mlb, pga] = await Promise.all([getMlbHomeRunFeed(), getPgaNormalizedMarkets()]);
  return {
    generatedAt: mlb.generatedAt,
    predictions: [...mlb.predictions, ...pga],
    gaps: [...mlb.gaps],
  };
}
