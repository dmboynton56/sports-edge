import type { Prediction } from "@/lib/data/types";
import { getSupabaseMissingEnv, getSupabaseRuntimeConfig } from "@/lib/data/supabase";

export type NflAnytimeTdRow = {
  id: string;
  game_id: string;
  season: number;
  week: number;
  game_date: string;
  game_time_utc: string;
  player_id: string;
  player_name: string;
  team: string;
  opponent: string;
  position: string;
  td_probability: number;
  sample_games: number;
  model_version: string;
  prediction_ts: string;
  quality_flags: string[] | null;
  best_book: string | null;
  best_book_title: string | null;
  best_price: number | null;
  market_probability: number | null;
  edge: number | null;
  ev: number | null;
  quarter_kelly: number | null;
  odds_snapshot_ts: string | null;
  odds_status: "priced" | "stale" | "missing";
};

export type NflAnytimeTdFeed = {
  generatedAt: string | null;
  predictions: Prediction[];
  gaps: string[];
};

const BLOCKING_QUALITY_FLAGS = new Set([
  "questionable",
  "secondary_depth_role",
  "deep_depth_chart",
  "roster_role_unverified",
  "limited_history",
  "missing_game_total",
]);
const MAX_RECOMMENDATION_PRICE = 1000;

async function supabaseRest<T>(resource: string): Promise<T[] | null> {
  const config = getSupabaseRuntimeConfig();
  if (!config.url || !config.anonKey) return null;
  const response = await fetch(`${config.url.replace(/\/$/, "")}/rest/v1/${resource}`, {
    headers: {
      apikey: config.anonKey,
      Authorization: `Bearer ${config.anonKey}`,
    },
    next: { revalidate: 60 },
  });
  if (!response.ok) return null;
  // SAFETY: The query selects the typed view contract above.
  return (await response.json()) as T[];
}

export function isQualifiedAnytimeTdRow(row: NflAnytimeTdRow) {
  const flags = Array.isArray(row.quality_flags) ? row.quality_flags : [];
  return row.odds_status === "priced"
    && row.best_price != null
    && row.best_price <= MAX_RECOMMENDATION_PRICE
    && row.sample_games >= 10
    && !flags.some((flag) => BLOCKING_QUALITY_FLAGS.has(flag));
}

export function mapAnytimeTdRow(row: NflAnytimeTdRow): Prediction {
  const historyConfidence = Math.min(1, row.sample_games / 50);
  return {
    id: `nfl-td-${row.game_id}-${row.player_id}`,
    sport: "NFL",
    league: "NFL",
    gameId: row.game_id,
    eventTime: row.game_time_utc,
    subject: `${row.player_name} TD (${row.team} vs ${row.opponent})`,
    player: row.player_name,
    market: "anytime_td",
    book: row.best_book_title ?? row.best_book ?? "n/a",
    line: null,
    price: row.best_price,
    modelProbability: row.td_probability,
    impliedProbability: row.market_probability,
    edge: row.edge,
    ev: row.ev,
    kelly: row.quarter_kelly,
    confidence: Math.min(0.9, 0.55 + 0.35 * historyConfidence),
    modelVersion: row.model_version,
    marketStatus: "research",
    detailHref: `/markets/nfl/${row.game_id}`,
    source: "Calibrated nflverse player model + The Odds API best price",
    updatedAt: row.odds_snapshot_ts ?? row.prediction_ts,
  };
}

export async function getNflAnytimeTdFeed(): Promise<NflAnytimeTdFeed> {
  const rows = await supabaseRest<NflAnytimeTdRow>(
    "nfl_anytime_td_edges_latest?select=*&order=ev.desc.nullslast&limit=500",
  );
  if (!rows) {
    const missing = getSupabaseMissingEnv();
    return {
      generatedAt: null,
      predictions: [],
      gaps: [
        missing.length
          ? `NFL anytime-TD feed unavailable: missing ${missing.join(", ")}.`
          : "NFL anytime-TD serving query failed.",
      ],
    };
  }

  const priced = rows.filter((row) => row.odds_status === "priced" && row.best_price != null);
  const qualified = priced.filter(isQualifiedAnytimeTdRow);
  const filtered = priced.length - qualified.length;
  const missingOdds = rows.length - priced.length;
  const timestamps = rows
    .flatMap((row) => [row.prediction_ts, row.odds_snapshot_ts])
    .filter((value): value is string => Boolean(value))
    .sort();
  return {
    generatedAt: timestamps.at(-1) ?? null,
    predictions: qualified.map(mapAnytimeTdRow),
    gaps: [
      "NFL anytime-TD probabilities passed a 2025 out-of-time outcome holdout, but sportsbook ROI is not yet backtested; one-way prices use raw implied probability rather than no-vig probability.",
      filtered
        ? `${filtered} priced anytime-TD rows are withheld by role, injury, sample-size, or +1000 longshot guardrails.`
        : "",
      missingOdds ? `${missingOdds} modeled NFL players do not have a current anytime-TD price.` : "",
    ].filter(Boolean),
  };
}
