// MLB Research Markets data fetchers
//
// These functions fetch predictions for research-labeled MLB markets:
// - Moneyline (v3)
// - Run line (home -1.5, v1)
// - Totals (over/under, v1)
//
// All markets are fail-closed: when sportsbook prices are missing, rows show
// model probabilities only with no edge/EV calculations.

import { getSupabaseRuntimeConfig, getSupabaseMissingEnv } from "@/lib/data/supabase";

const MLB_SLATE_TIME_ZONE = "America/Denver";

type SupabaseResearchRow = {
  prediction_id: string;
  league: string;
  market: string;
  model_version: string;
  model_status: string;
  game_id: string;
  game_pk: number;
  season: number;
  game_date: string;
  game_datetime: string | null;
  home_team: string;
  away_team: string;
  venue: string | null;
  as_of_ts: string;
  
  // Moneyline
  home_win_prob: number | null;
  away_win_prob: number | null;
  
  // Run-line
  p_home_cover_15: number | null;
  p_away_cover_plus_15: number | null;
  
  // Totals
  predicted_total: number | null;
  p_over_8_5: number | null;
  p_over_9_5: number | null;
  
  // Odds
  odds_status: string;
  odds_snapshot_ts: string | null;
  best_book: string | null;
  home_price: number | null;
  away_price: number | null;
  home_runline_price: number | null;
  away_runline_price: number | null;
  total_line: number | null;
  over_price: number | null;
  under_price: number | null;
  
  // Edge
  implied_probability: number | null;
  no_vig_probability: number | null;
  edge: number | null;
  ev: number | null;
  kelly: number | null;
};

export type MlbResearchPrediction = {
  id: string;
  market: "moneyline" | "run_line" | "total";
  modelVersion: string;
  gameId: string;
  gamePk: number;
  gameDate: string;
  homeTeam: string;
  awayTeam: string;
  venue: string | null;
  
  // Moneyline
  homeWinProb?: number | null;
  awayWinProb?: number | null;
  
  // Run-line
  pHomeCover15?: number | null;
  pAwayCoverPlus15?: number | null;
  
  // Totals
  predictedTotal?: number | null;
  pOver85?: number | null;
  pOver95?: number | null;
  
  // Odds
  oddsStatus: "ok" | "missing_odds" | "stale";
  bestBook?: string | null;
  homePrice?: number | null;
  awayPrice?: number | null;
  homeRunlinePrice?: number | null;
  awayRunlinePrice?: number | null;
  totalLine?: number | null;
  overPrice?: number | null;
  underPrice?: number | null;
  
  // Edge (null when oddsStatus !== 'ok')
  edge?: number | null;
  ev?: number | null;
  kelly?: number | null;
  
  asOfTs: string;
};

export type MlbResearchBoardData = {
  generatedAt: string | null;
  market: "moneyline" | "run_line" | "total";
  slateDate: string;
  predictions: MlbResearchPrediction[];
  gaps: string[];
  dataSource: "supabase" | "unavailable";
};

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

function mapResearchRow(row: SupabaseResearchRow): MlbResearchPrediction {
  const base = {
    id: row.prediction_id,
    market: row.market as "moneyline" | "run_line" | "total",
    modelVersion: row.model_version,
    gameId: row.game_id,
    gamePk: row.game_pk,
    gameDate: row.game_date,
    homeTeam: row.home_team,
    awayTeam: row.away_team,
    venue: row.venue,
    oddsStatus: row.odds_status as "ok" | "missing_odds" | "stale",
    bestBook: row.best_book,
    edge: row.edge,
    ev: row.ev,
    kelly: row.kelly,
    asOfTs: row.as_of_ts,
  };

  if (row.market === "moneyline") {
    return {
      ...base,
      homeWinProb: row.home_win_prob,
      awayWinProb: row.away_win_prob,
      homePrice: row.home_price,
      awayPrice: row.away_price,
    };
  }

  if (row.market === "run_line") {
    return {
      ...base,
      pHomeCover15: row.p_home_cover_15,
      pAwayCoverPlus15: row.p_away_cover_plus_15,
      homeRunlinePrice: row.home_runline_price,
      awayRunlinePrice: row.away_runline_price,
    };
  }

  // total
  return {
    ...base,
    predictedTotal: row.predicted_total,
    pOver85: row.p_over_8_5,
    pOver95: row.p_over_9_5,
    totalLine: row.total_line,
    overPrice: row.over_price,
    underPrice: row.under_price,
  };
}

function supabaseConfigGaps(): string[] {
  const missingEnv = getSupabaseMissingEnv();
  return missingEnv.length
    ? [`Supabase live feed unavailable: missing ${missingEnv.join(", ")}.`]
    : [];
}

export async function getMlbResearchBoard(
  market: "moneyline" | "run_line" | "total"
): Promise<MlbResearchBoardData> {
  const slateDate = todayInTimeZone(MLB_SLATE_TIME_ZONE);
  const rows = await supabaseRest<SupabaseResearchRow>(
    `mlb_research_predictions_latest?select=*&game_date=eq.${slateDate}&market=eq.${market}&order=game_pk.asc&limit=500`
  );

  if (!rows || rows.length === 0) {
    return {
      generatedAt: null,
      market,
      slateDate,
      predictions: [],
      gaps: [
        ...supabaseConfigGaps(),
        `No MLB ${market} research predictions available for ${slateDate}.`,
      ],
      dataSource: rows === null ? "unavailable" : "supabase",
    };
  }

  const predictions = rows.map(mapResearchRow);
  const generatedAt = rows[0]?.as_of_ts ?? null;
  const missingOdds = rows.filter((r) => r.odds_status === "missing_odds").length;

  return {
    generatedAt,
    market,
    slateDate,
    predictions,
    gaps: [
      ...supabaseConfigGaps(),
      missingOdds
        ? `${missingOdds} games do not have sportsbook prices (showing model probabilities only).`
        : null,
    ].filter((g): g is string => Boolean(g)),
    dataSource: "supabase",
  };
}
