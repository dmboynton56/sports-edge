import { getSupabaseMissingEnv, supabaseRest } from "@/lib/data/supabase";
import type { JsonObject } from "@/lib/data/json";

export type FeatureDriver = {
  feature: string;
  value: number;
  impact: number;
  isHeuristic?: boolean;
};

export type GameExplanation = {
  gameId: string;
  league: string;
  modelVersion: string;
  predictionTs: string;
  topFeatures: FeatureDriver[];
  injuryAdjusted: boolean;
  homeInjuryDelta: number | null;
  awayInjuryDelta: number | null;
  baseVsAdjusted: JsonObject | null;
};

type SupabaseExplanationRow = {
  game_id: string;
  league: string;
  model_version: string;
  prediction_ts: string;
  top_features: FeatureDriver[] | string;
  injury_adjusted: boolean;
  home_injury_delta: number | null;
  away_injury_delta: number | null;
  base_vs_adjusted: JsonObject | null;
};

function parseTopFeatures(raw: FeatureDriver[] | string): FeatureDriver[] {
  if (Array.isArray(raw)) return raw;
  try {
    // SAFETY: The database column is written by the explanation pipeline as a FeatureDriver array.
    const parsed = JSON.parse(raw) as FeatureDriver[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function mapRow(row: SupabaseExplanationRow): GameExplanation {
  return {
    gameId: row.game_id,
    league: row.league,
    modelVersion: row.model_version,
    predictionTs: row.prediction_ts,
    topFeatures: parseTopFeatures(row.top_features),
    injuryAdjusted: Boolean(row.injury_adjusted),
    homeInjuryDelta: row.home_injury_delta,
    awayInjuryDelta: row.away_injury_delta,
    baseVsAdjusted: row.base_vs_adjusted,
  };
}

export async function getGameExplanation(
  gameId: string,
  league: "NBA" | "NFL",
): Promise<GameExplanation | null> {
  const missing = getSupabaseMissingEnv();
  if (missing.length) return null;

  const resource =
    `game_explanations?game_id=eq.${gameId}&league=eq.${league}` +
    `&order=prediction_ts.desc&limit=1`;
  const rows = await supabaseRest<SupabaseExplanationRow>(resource, 60);
  const row = rows?.[0];
  return row ? mapRow(row) : null;
}
