import { getSupabaseMissingEnv, getSupabaseRuntimeConfig } from "@/lib/data/supabase";

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
  baseVsAdjusted: Record<string, unknown> | null;
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
  base_vs_adjusted: Record<string, unknown> | null;
};

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

function parseTopFeatures(raw: FeatureDriver[] | string): FeatureDriver[] {
  if (Array.isArray(raw)) return raw;
  try {
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
  const rows = await supabaseRest<SupabaseExplanationRow>(resource);
  const row = rows?.[0];
  return row ? mapRow(row) : null;
}
