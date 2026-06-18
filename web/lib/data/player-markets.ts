import { promises as fs } from "fs";
import path from "path";

import type { Prediction } from "@/lib/data/types";
import { getSupabaseRuntimeConfig } from "@/lib/data/supabase";

export type MlbHomeRunPrediction = Prediction & {
  gameDate?: string | null;
  team?: string | null;
  opponent?: string | null;
  venue?: string | null;
  lineupSlot?: number | null;
  lineupStatus?: string | null;
  opposingProbablePitcher?: string | null;
  baselineProbability?: number | null;
  rank?: number | null;
  qualityFlags?: string[];
  topFeatures?: { feature: string; value: number }[];
};

export type MlbHomeRunFeed = {
  generatedAt: string | null;
  modelVersion: string;
  productionStatus: "candidate" | "approved" | "blocked";
  predictions: MlbHomeRunPrediction[];
  gaps: string[];
};

const MLB_HR_PATH = path.join(process.cwd(), "public", "data", "mlb_home_runs.json");
const MLB_SLATE_TIME_ZONE = "America/Denver";
const PGA_TOURNAMENT_PATH = path.join(
  process.cwd(),
  "public",
  "data",
  "pga_tournaments",
  "us_open_2026.json",
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
  rank: number | null;
  confidence: number | null;
  model_version: string;
  prediction_ts: string | null;
  quality_flags: string[] | null;
  top_features: { feature: string; value: number }[] | null;
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
    rank: row.rank,
    qualityFlags: row.quality_flags ?? [],
    topFeatures: row.top_features ?? [],
  };
}

export async function getMlbHomeRunFeed(): Promise<MlbHomeRunFeed> {
  const slateDate = todayInTimeZone(MLB_SLATE_TIME_ZONE);
  const rows = await supabaseRest<SupabaseMlbHrRow>(
    `mlb_home_run_predictions_latest?select=*&game_date=eq.${slateDate}&order=rank.asc&limit=120`,
  );
  if (rows && rows.length) {
    return {
      generatedAt: rows[0]?.prediction_ts ?? null,
      modelVersion: rows[0]?.model_version ?? "mlb-hr-v1-heuristic",
      productionStatus: "candidate",
      predictions: rows.map(mapSupabaseMlb),
      gaps: [],
    };
  }

  try {
    const payload = JSON.parse(await fs.readFile(MLB_HR_PATH, "utf8")) as MlbHomeRunFeed;
    const predictions = (payload.predictions ?? []).filter(
      (row) => predictionGameDate(row) === slateDate,
    );
    const existingGaps = payload.gaps ?? [];
    return {
      ...payload,
      predictions,
      gaps: predictions.length
        ? existingGaps
        : [
            ...existingGaps,
            `No MLB home run predictions available for ${slateDate}.`,
          ],
    };
  } catch {
    return {
      generatedAt: null,
      modelVersion: "mlb-hr-v1-heuristic",
      productionStatus: "candidate",
      predictions: [],
      gaps: ["No MLB home run artifact found at web/public/data/mlb_home_runs.json."],
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
