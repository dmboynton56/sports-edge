import { getSupabaseMissingEnv, getSupabaseRuntimeConfig } from "@/lib/data/supabase";

export type FreshnessStatus = "fresh" | "stale" | "no_prediction" | "no_odds";

export type TeamSlateGame = {
  gameId: string;
  league: string;
  season: number | null;
  week: number | null;
  gameTimeUtc: string;
  gameDate: string | null;
  homeTeam: string;
  awayTeam: string;
  bookSpread: number | null;
  modelSpread: number | null;
  edgePts: number | null;
  homeWinProb: number | null;
  modelVersion: string | null;
  predictionTs: string | null;
  oddsTs: string | null;
  freshnessStatus: FreshnessStatus;
  injuryAdjusted?: boolean;
  injuryDataMissing?: boolean;
};

export type TeamSlateFeed = {
  league: string;
  generatedAt: string;
  windowStart: string;
  windowEnd: string;
  games: TeamSlateGame[];
  gaps: string[];
};

const SLATE_TIME_ZONE = "America/Denver";
const FRESH_HOURS = 24;

type SupabaseGameRow = {
  id: string;
  league: string;
  season: number | null;
  week: number | null;
  game_time_utc: string;
  game_date: string | null;
  home_team: string;
  away_team: string;
  book_spread: number | null;
};

type SupabasePredictionRow = {
  game_id: string;
  my_spread: number | null;
  my_home_win_prob: number | null;
  model_version: string | null;
  asof_ts: string;
};

type SupabaseOddsRow = {
  game_id: string;
  line: number | null;
  snapshot_ts: string;
  market: string;
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

function addDays(isoDate: string, days: number): string {
  const date = new Date(`${isoDate}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

function gameServingDate(row: SupabaseGameRow): string {
  if (row.game_date) return row.game_date.slice(0, 10);
  return new Date(row.game_time_utc).toLocaleDateString("en-CA", {
    timeZone: SLATE_TIME_ZONE,
  });
}

function deriveFreshness(
  predictionTs: string | null,
  bookSpread: number | null,
): FreshnessStatus {
  if (!predictionTs) return "no_prediction";
  const ageMs = Date.now() - new Date(predictionTs).getTime();
  if (ageMs > FRESH_HOURS * 60 * 60 * 1000) return "stale";
  if (bookSpread == null) return "no_odds";
  return "fresh";
}

function supabaseConfigGaps(): string[] {
  const missingEnv = getSupabaseMissingEnv();
  return missingEnv.length
    ? [`Supabase live feed unavailable: missing ${missingEnv.join(", ")}.`]
    : [];
}

async function fetchGamesInWindow(league: string, start: string, end: string) {
  const resource =
    `games?league=eq.${league}` +
    `&game_date=gte.${start}&game_date=lte.${end}` +
    `&order=game_time_utc.asc` +
    `&select=id,league,season,week,game_time_utc,game_date,home_team,away_team,book_spread`;
  return supabaseRest<SupabaseGameRow>(resource);
}

async function fetchLatestPredictions(gameIds: string[]) {
  if (!gameIds.length) return new Map<string, SupabasePredictionRow>();
  const inList = gameIds.map((id) => `"${id}"`).join(",");
  const resource =
    `model_predictions?game_id=in.(${inList})` +
    `&order=asof_ts.desc` +
    `&select=game_id,my_spread,my_home_win_prob,model_version,asof_ts`;
  const rows = (await supabaseRest<SupabasePredictionRow>(resource)) ?? [];
  const latest = new Map<string, SupabasePredictionRow>();
  for (const row of rows) {
    if (!latest.has(row.game_id)) latest.set(row.game_id, row);
  }
  return latest;
}

async function fetchLatestOdds(gameIds: string[]) {
  if (!gameIds.length) return new Map<string, SupabaseOddsRow>();
  const inList = gameIds.map((id) => `"${id}"`).join(",");
  const resource =
    `odds_snapshots?game_id=in.(${inList})&market=eq.spread` +
    `&order=snapshot_ts.desc` +
    `&select=game_id,line,snapshot_ts,market`;
  const rows = (await supabaseRest<SupabaseOddsRow>(resource)) ?? [];
  const latest = new Map<string, SupabaseOddsRow>();
  for (const row of rows) {
    if (!latest.has(row.game_id)) latest.set(row.game_id, row);
  }
  return latest;
}

function buildSlateGame(
  game: SupabaseGameRow,
  prediction: SupabasePredictionRow | undefined,
  odds: SupabaseOddsRow | undefined,
): TeamSlateGame {
  const bookSpread = game.book_spread ?? odds?.line ?? null;
  const modelSpread = prediction?.my_spread ?? null;
  const edgePts =
    modelSpread != null && bookSpread != null ? modelSpread - bookSpread : null;

  return {
    gameId: game.id,
    league: game.league,
    season: game.season,
    week: game.week,
    gameTimeUtc: game.game_time_utc,
    gameDate: gameServingDate(game),
    homeTeam: game.home_team,
    awayTeam: game.away_team,
    bookSpread,
    modelSpread,
    edgePts,
    homeWinProb: prediction?.my_home_win_prob ?? null,
    modelVersion: prediction?.model_version ?? null,
    predictionTs: prediction?.asof_ts ?? null,
    oddsTs: odds?.snapshot_ts ?? null,
    freshnessStatus: deriveFreshness(prediction?.asof_ts ?? null, bookSpread),
  };
}

export async function getTeamSlateFeed(
  league: "NBA" | "NFL",
  options?: { lookaheadDays?: number },
): Promise<TeamSlateFeed> {
  const gaps = supabaseConfigGaps();
  const today = todayInTimeZone(SLATE_TIME_ZONE);
  const lookahead = options?.lookaheadDays ?? (league === "NFL" ? 7 : 1);
  const windowStart = today;
  const windowEnd = addDays(today, lookahead);

  const games = await fetchGamesInWindow(league, windowStart, windowEnd);
  if (!games) {
    return {
      league,
      generatedAt: new Date().toISOString(),
      windowStart,
      windowEnd,
      games: [],
      gaps: gaps.length ? gaps : ["Supabase games query failed."],
    };
  }

  const gameIds = games.map((g) => g.id);
  const [predictions, oddsRows] = await Promise.all([
    fetchLatestPredictions(gameIds),
    fetchLatestOdds(gameIds),
  ]);

  const slateGames = games.map((game) =>
    buildSlateGame(game, predictions.get(game.id), oddsRows.get(game.id)),
  );

  const resultGaps = [...gaps];
  if (!slateGames.length) {
    resultGaps.push(`No ${league} games in window ${windowStart} to ${windowEnd}.`);
  }
  const missingPreds = slateGames.filter((g) => g.freshnessStatus === "no_prediction").length;
  if (slateGames.length && missingPreds === slateGames.length) {
    resultGaps.push(`All ${league} games missing predictions.`);
  }

  return {
    league,
    generatedAt: new Date().toISOString(),
    windowStart,
    windowEnd,
    games: slateGames,
    gaps: resultGaps,
  };
}

export async function getTeamSlateGame(
  league: "NBA" | "NFL",
  gameId: string,
): Promise<TeamSlateGame | null> {
  const config = getSupabaseRuntimeConfig();
  if (!config.url || !config.anonKey) return null;

  const resource =
    `games?id=eq.${gameId}&league=eq.${league}` +
    `&select=id,league,season,week,game_time_utc,game_date,home_team,away_team,book_spread`;
  const games = await supabaseRest<SupabaseGameRow>(resource);
  const game = games?.[0];
  if (!game) return null;

  const [predictions, oddsRows] = await Promise.all([
    fetchLatestPredictions([game.id]),
    fetchLatestOdds([game.id]),
  ]);

  return buildSlateGame(game, predictions.get(game.id), oddsRows.get(game.id));
}
