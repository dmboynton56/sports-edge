import { promises as fs } from "fs";
import path from "path";

import type { Prediction } from "@/lib/data/types";

type RawPrediction = Partial<Prediction> & {
  game_id?: string;
  event_time?: string | null;
  model_probability?: number | null;
  implied_probability?: number | null;
  model_version?: string;
};

type PredictionPayload = {
  generatedAt?: string | null;
  generated_at?: string | null;
  predictions?: RawPrediction[];
  gaps?: string[];
};

const PREDICTIONS_PATH = path.join(process.cwd(), "public", "data", "predictions.json");

function normalizePrediction(raw: RawPrediction, index: number): Prediction {
  const sport = raw.sport ?? raw.league ?? "Unknown";
  const subject =
    raw.subject ??
    raw.player ??
    [raw.awayTeam, raw.homeTeam].filter(Boolean).join(" @ ") ??
    "n/a";

  return {
    id: raw.id ?? `${sport}-${raw.gameId ?? raw.game_id ?? index}`,
    sport,
    league: raw.league ?? sport,
    gameId: raw.gameId ?? raw.game_id ?? "n/a",
    eventTime: raw.eventTime ?? raw.event_time ?? null,
    subject,
    homeTeam: raw.homeTeam ?? null,
    awayTeam: raw.awayTeam ?? null,
    player: raw.player ?? null,
    market: raw.market ?? "n/a",
    book: raw.book ?? "n/a",
    line: typeof raw.line === "number" ? raw.line : null,
    price: typeof raw.price === "number" ? raw.price : null,
    modelProbability:
      typeof raw.modelProbability === "number"
        ? raw.modelProbability
        : raw.model_probability ?? null,
    impliedProbability:
      typeof raw.impliedProbability === "number"
        ? raw.impliedProbability
        : raw.implied_probability ?? null,
    edge: typeof raw.edge === "number" ? raw.edge : null,
    ev: typeof raw.ev === "number" ? raw.ev : null,
    kelly: typeof raw.kelly === "number" ? raw.kelly : null,
    confidence: typeof raw.confidence === "number" ? raw.confidence : null,
    modelVersion: raw.modelVersion ?? raw.model_version ?? "n/a",
    source: raw.source,
    updatedAt: raw.updatedAt ?? null,
  };
}

export async function getLocalPredictions(): Promise<{
  generatedAt: string | null;
  predictions: Prediction[];
  gaps: string[];
}> {
  try {
    const payload = JSON.parse(await fs.readFile(PREDICTIONS_PATH, "utf8")) as PredictionPayload;
    return {
      generatedAt: payload.generatedAt ?? payload.generated_at ?? null,
      predictions: (payload.predictions ?? []).map(normalizePrediction),
      gaps: payload.gaps ?? [],
    };
  } catch {
    return {
      generatedAt: null,
      predictions: [],
      gaps: ["No local prediction export found at web/public/data/predictions.json."],
    };
  }
}
