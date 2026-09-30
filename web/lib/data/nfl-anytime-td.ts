import type { Prediction } from "@/lib/data/types";
import { getSupabaseMissingEnv, supabaseRest } from "@/lib/data/supabase";
import { denverDate, nflPredictionIsCurrent, nflWeekWindow } from "@/lib/nfl-week";

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

export function isQualifiedAnytimeTdRow(row: NflAnytimeTdRow) {
  const flags = Array.isArray(row.quality_flags) ? row.quality_flags : [];
  return row.td_probability > 0
    && row.td_probability < 1
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
    book: "model",
    line: null,
    price: null,
    modelProbability: row.td_probability,
    impliedProbability: null,
    edge: null,
    ev: null,
    kelly: null,
    confidence: Math.min(0.9, 0.55 + 0.35 * historyConfidence),
    modelVersion: row.model_version,
    marketStatus: "model_only",
    detailHref: `/markets/nfl/${row.game_id}`,
    source: "Calibrated nflverse player model; fair odds derived from model probability",
    updatedAt: row.prediction_ts,
  };
}

export async function getNflAnytimeTdFeed(): Promise<NflAnytimeTdFeed> {
  const rows = await supabaseRest<NflAnytimeTdRow>(
    "nfl_anytime_td_edges_latest?select=*&order=td_probability.desc&limit=500",
    60,
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

  const today = denverDate();
  const end = nflWeekWindow(today).end;
  const qualified = rows.filter((row) => row.game_date >= today && row.game_date <= end
    && nflPredictionIsCurrent(row.prediction_ts) && isQualifiedAnytimeTdRow(row));
  const filtered = rows.length - qualified.length;
  const timestamps = qualified
    .map((row) => row.prediction_ts)
    .filter((value): value is string => Boolean(value))
    .sort();
  return {
    generatedAt: timestamps.at(-1) ?? null,
    predictions: qualified.map(mapAnytimeTdRow),
    gaps: [
      "NFL anytime-TD fair odds come from model probabilities. They are not sportsbook offers, and no betting edge or EV is calculated.",
      filtered
        ? `${filtered} NFL anytime-TD rows are withheld by forecast freshness, game window, role, injury, sample-size, or invalid probability guardrails.`
        : "",
    ].filter(Boolean),
  };
}
