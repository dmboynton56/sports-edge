import { getSupabaseMissingEnv, getSupabaseRuntimeConfig } from "@/lib/data/supabase";
import type { Prediction } from "@/lib/data/types";

type SupabaseCfbPrediction = {
  event_id: string;
  game_time_utc: string;
  game_date: string;
  home_team: string;
  away_team: string;
  venue: string | null;
  neutral_site: boolean;
  model_version: string;
  model_status: string;
  predicted_home_points: number;
  predicted_away_points: number;
  predicted_margin: number;
  predicted_total: number;
  home_win_probability: number;
  confidence: number | null;
  quality_flags: string[] | null;
  prediction_ts: string;
};

type SupabaseCfbMarket = SupabaseCfbPrediction & {
  market: "moneyline" | "spread" | "total";
  selection: "home" | "away" | "over" | "under";
  subject: string;
  book: string;
  book_title: string | null;
  line: number | null;
  price: number;
  model_probability: number;
  implied_probability: number;
  edge: number;
  ev: number;
  quarter_kelly: number | null;
  odds_snapshot_ts: string;
  odds_status: "priced" | "stale";
};

export type CfbSlateGame = {
  eventId: string;
  eventTime: string;
  gameDate: string;
  homeTeam: string;
  awayTeam: string;
  venue: string | null;
  predictedHomePoints: number;
  predictedAwayPoints: number;
  predictedMargin: number;
  predictedTotal: number;
  homeWinProbability: number;
  confidence: number | null;
  modelVersion: string;
};

export type CfbMarketFeed = {
  generatedAt: string | null;
  games: CfbSlateGame[];
  predictions: Prediction[];
  gaps: string[];
};

async function supabaseRest<T>(resource: string): Promise<T[] | null> {
  const config = getSupabaseRuntimeConfig();
  if (!config.url || !config.anonKey) return null;
  const response = await fetch(`${config.url.replace(/\/$/, "")}/rest/v1/${resource}`, {
    headers: { apikey: config.anonKey, Authorization: `Bearer ${config.anonKey}` },
    next: { revalidate: 60 },
  });
  if (!response.ok) return null;
  return (await response.json()) as T[];
}

function mapGame(row: SupabaseCfbPrediction): CfbSlateGame {
  return {
    eventId: row.event_id,
    eventTime: row.game_time_utc,
    gameDate: row.game_date,
    homeTeam: row.home_team,
    awayTeam: row.away_team,
    venue: row.venue,
    predictedHomePoints: row.predicted_home_points,
    predictedAwayPoints: row.predicted_away_points,
    predictedMargin: row.predicted_margin,
    predictedTotal: row.predicted_total,
    homeWinProbability: row.home_win_probability,
    confidence: row.confidence,
    modelVersion: row.model_version,
  };
}

function mapMarket(row: SupabaseCfbMarket): Prediction {
  return {
    id: `${row.event_id}-${row.market}-${row.selection}`,
    sport: "CFB",
    league: "CFB",
    gameId: row.event_id,
    eventTime: row.game_time_utc,
    subject: row.subject,
    homeTeam: row.home_team,
    awayTeam: row.away_team,
    market: row.market,
    book: row.book_title ?? row.book,
    line: row.line,
    price: row.price,
    modelProbability: row.model_probability,
    impliedProbability: row.implied_probability,
    edge: row.odds_status === "priced" ? row.edge : null,
    ev: row.odds_status === "priced" ? row.ev : null,
    kelly: row.odds_status === "priced" ? row.quarter_kelly : null,
    confidence: row.confidence,
    modelVersion: row.model_version,
    source: "Supabase cfb_market_edges_latest",
    updatedAt: row.odds_snapshot_ts,
  };
}

export async function getCfbMarketFeed(): Promise<CfbMarketFeed> {
  const [gameRows, marketRows] = await Promise.all([
    supabaseRest<SupabaseCfbPrediction>(
      "cfb_team_predictions_latest?select=*&order=game_time_utc.asc&limit=500",
    ),
    supabaseRest<SupabaseCfbMarket>(
      "cfb_market_edges_latest?select=*&order=game_time_utc.asc,market.asc,edge.desc&limit=1500",
    ),
  ]);
  const games = (gameRows ?? []).map(mapGame);
  const predictions = (marketRows ?? []).map(mapMarket);
  const stale = (marketRows ?? []).filter((row) => row.odds_status === "stale").length;
  const moneylineGames = new Set(
    (marketRows ?? []).filter((row) => row.market === "moneyline").map((row) => row.event_id),
  ).size;
  const missing = getSupabaseMissingEnv();
  return {
    generatedAt: (gameRows ?? []).map((row) => row.prediction_ts).sort().at(-1) ?? null,
    games,
    predictions,
    gaps: [
      ...(missing.length ? [`Supabase live feed unavailable: missing ${missing.join(", ")}.`] : []),
      "CFB prices and EV are research-only: the model passed an out-of-time outcome holdout but has no historical closing-line backtest.",
      "V1 guardrail withholds prices above +400 and any row where absolute estimated edge exceeds 8% or absolute EV exceeds 20%.",
      moneylineGames < games.length
        ? `${games.length - moneylineGames} games have no guardrail-eligible moneyline signal; projected points and win chances remain available.`
        : null,
      stale ? `${stale} priced rows are older than 24 hours; edge and EV are withheld.` : null,
      gameRows === null || marketRows === null ? "The CFB serving views could not be reached." : null,
      games.length === 0 ? "No upcoming CFB slate is published." : null,
    ].filter((gap): gap is string => Boolean(gap)),
  };
}
