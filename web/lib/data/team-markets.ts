import { getSupabaseMissingEnv, getSupabaseRuntimeConfig, asRestRows } from "@/lib/data/supabase";
import type { Prediction } from "@/lib/data/types";

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
  book: string;
  line: number | null;
  price: number | null;
  snapshot_ts: string;
  market: string;
  selection: string | null;
};

type TeamMarketFeed = {
  generatedAt: string | null;
  predictions: Prediction[];
  gaps: string[];
};

// Held-out residual scales used only to translate a point projection into a
// research cover probability. They are deliberately surfaced as approximate,
// not as a promoted betting model.
const SPREAD_RESIDUAL_SIGMA = {
  NFL: 13.957504133352113,
  NBA: 15.191160518903473,
} satisfies Record<"NBA" | "NFL", number>;

async function supabaseRest<T>(resource: string): Promise<T[] | null> {
  const config = getSupabaseRuntimeConfig();
  if (!config.url || !config.anonKey) return null;
  try {
    const base = config.url.replace(/\/$/, "");
    const response = await fetch(`${base}/rest/v1/${resource}`, {
      headers: {
        apikey: config.anonKey,
        Authorization: `Bearer ${config.anonKey}`,
      },
      next: { revalidate: 60 },
    });
    if (!response.ok) return null;
    const payload = await response.json();
    return asRestRows<T>(payload);
  } catch {
    return null;
  }
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
    `odds_snapshots?game_id=in.(${inList})&market=eq.spread&selection=eq.home` +
    `&order=snapshot_ts.desc` +
    `&select=game_id,book,line,price,snapshot_ts,market,selection`;
  const rows = (await supabaseRest<SupabaseOddsRow>(resource)) ?? [];
  const latest = new Map<string, SupabaseOddsRow>();
  for (const row of rows) {
    if (!latest.has(row.game_id)) latest.set(row.game_id, row);
  }
  return latest;
}

async function fetchLatestFeaturedOdds(gameIds: string[]) {
  if (!gameIds.length) return [];
  const inList = gameIds.map((id) => `"${id}"`).join(",");
  const resource =
    `odds_snapshots?game_id=in.(${inList})&selection=not.is.null` +
    `&market=in.(moneyline,spread,total)` +
    `&order=snapshot_ts.desc` +
    `&select=game_id,book,line,price,snapshot_ts,market,selection`;
  const rows = (await supabaseRest<SupabaseOddsRow>(resource)) ?? [];
  const latest = new Map<string, SupabaseOddsRow>();
  for (const row of rows) {
    const key = `${row.game_id}:${row.market}:${row.selection}`;
    if (!latest.has(key)) latest.set(key, row);
  }
  return Array.from(latest.values());
}

export function americanImpliedProbability(price: number | null): number | null {
  if (price == null || price === 0) return null;
  return price > 0 ? 100 / (price + 100) : -price / (-price + 100);
}

function americanDecimalOdds(price: number | null): number | null {
  if (price == null || price === 0) return null;
  return price > 0 ? 1 + price / 100 : 1 + 100 / -price;
}

function normalCdf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * x);
  const erf = sign * (
    1 -
    (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x)
  );
  return 0.5 * (1 + erf);
}

export function spreadCoverProbability(
  modelSpread: number,
  offeredLine: number,
  selection: "home" | "away",
  sigma: number,
) {
  const modelMargin = -modelSpread;
  const selectedMargin = selection === "home" ? modelMargin : -modelMargin;
  return normalCdf((selectedMargin + offeredLine) / sigma);
}

function noVigImpliedProbability(price: number | null, counterpartPrice: number | null) {
  const raw = americanImpliedProbability(price);
  const counterpart = americanImpliedProbability(counterpartPrice);
  if (raw == null) return null;
  if (counterpart == null) return raw;
  return raw / (raw + counterpart);
}

function expectedValue(probability: number | null, price: number | null) {
  const decimalOdds = americanDecimalOdds(price);
  if (probability == null || decimalOdds == null) return null;
  return probability * decimalOdds - 1;
}

function quarterKelly(probability: number | null, price: number | null) {
  const decimalOdds = americanDecimalOdds(price);
  if (probability == null || decimalOdds == null) return null;
  const profitMultiple = decimalOdds - 1;
  if (profitMultiple <= 0) return null;
  return Math.max(0, ((profitMultiple * probability - (1 - probability)) / profitMultiple) * 0.25);
}

function counterpartSelection(market: string, selection: string | null) {
  if (market === "total") return selection === "over" ? "under" : "over";
  return selection === "home" ? "away" : "home";
}

export function buildTeamMarketPredictions(
  league: "NBA" | "NFL",
  games: SupabaseGameRow[],
  predictionRows: SupabasePredictionRow[],
  oddsRows: SupabaseOddsRow[],
): Prediction[] {
  const predictions = new Map(predictionRows.map((row) => [row.game_id, row]));
  const odds = new Map(
    oddsRows.map((row) => [`${row.game_id}:${row.market}:${row.selection}`, row]),
  );

  return games.flatMap((game) => {
    const prediction = predictions.get(game.id);
    const eventTime = game.game_time_utc;
    return oddsRows
      .filter((row) => row.game_id === game.id)
      .map((row): Prediction => {
        const selection = row.selection ?? "unknown";
        const counterpart = odds.get(
          `${row.game_id}:${row.market}:${counterpartSelection(row.market, row.selection)}`,
        );
        const impliedProbability = noVigImpliedProbability(row.price, counterpart?.price ?? null);
        let modelProbability: number | null = null;
        let modelVersion = "unmodeled";

        if (prediction && row.market === "moneyline") {
          modelProbability = selection === "home"
            ? prediction.my_home_win_prob
            : prediction.my_home_win_prob == null
              ? null
              : 1 - prediction.my_home_win_prob;
          modelVersion = prediction.model_version ?? "unknown";
        } else if (
          prediction &&
          row.market === "spread" &&
          prediction.my_spread != null &&
          row.line != null &&
          (selection === "home" || selection === "away")
        ) {
          modelProbability = spreadCoverProbability(
            prediction.my_spread,
            row.line,
            selection,
            SPREAD_RESIDUAL_SIGMA[league],
          );
          modelVersion = `${prediction.model_version ?? "unknown"}-residual`;
        }

        const edge = modelProbability != null && impliedProbability != null
          ? modelProbability - impliedProbability
          : null;
        const subjectTeam = selection === "home" ? game.home_team : game.away_team;
        const subject = row.market === "total"
          ? `${game.away_team} @ ${game.home_team} ${selection}`
          : `${subjectTeam} ${row.market}`;

        return {
          id: `${game.id}-${row.market}-${selection}`,
          sport: league,
          league,
          gameId: game.id,
          eventTime,
          subject,
          homeTeam: game.home_team,
          awayTeam: game.away_team,
          market: row.market,
          book: row.book,
          line: row.line,
          price: row.price,
          modelProbability,
          impliedProbability,
          edge,
          ev: expectedValue(modelProbability, row.price),
          kelly: quarterKelly(modelProbability, row.price),
          confidence: modelProbability == null ? null : Math.abs(modelProbability - 0.5) * 2,
          modelVersion,
          marketStatus: modelProbability == null ? "model_only" : "research",
          detailHref: `/markets/${league.toLowerCase()}/${game.id}`,
          source: "Supabase team model + The Odds API snapshot",
          updatedAt: prediction?.asof_ts ?? row.snapshot_ts,
        };
      });
  });
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
  const lookahead = options?.lookaheadDays ?? (league === "NFL" ? 14 : 1);
  const windowStart = today;
  const windowEnd = addDays(today, lookahead);

  const games = await fetchGamesInWindow(league, windowStart, windowEnd);
  if (!Array.isArray(games)) {
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

export async function getTeamMarketPredictions(
  league: "NBA" | "NFL",
  options?: { lookaheadDays?: number },
): Promise<TeamMarketFeed> {
  const today = todayInTimeZone(SLATE_TIME_ZONE);
  const lookaheadDays = options?.lookaheadDays ?? (league === "NFL" ? 14 : 2);
  const end = addDays(today, lookaheadDays);
  try {
    const games = await fetchGamesInWindow(league, today, end);
    if (!Array.isArray(games) || games.length === 0) {
      return {
        generatedAt: null,
        predictions: [],
        gaps: Array.isArray(games)
          ? [`No ${league} games in window ${today} to ${end}.`]
          : [...supabaseConfigGaps(), `Supabase ${league} game query failed.`],
      };
    }

    const gameIds = games.map((game) => game.id);
    const [predictionMap, oddsRows] = await Promise.all([
      fetchLatestPredictions(gameIds),
      fetchLatestFeaturedOdds(gameIds),
    ]);
    const predictionRows = Array.from(predictionMap.values());
    const normalized = buildTeamMarketPredictions(league, games, predictionRows, oddsRows);
    const gaps: string[] = [];
    if (predictionRows.length < games.length) {
      gaps.push(`${games.length - predictionRows.length} ${league} games lack a current team prediction.`);
    }
    if (oddsRows.length < games.length * 6) {
      gaps.push(`${league} featured-market outcome coverage is incomplete (${oddsRows.length}/${games.length * 6}).`);
    }
    if (normalized.some((row) => row.market === "total")) {
      gaps.push(`${league} totals prices are live, but a validated totals model head is not yet available; edge and EV stay blank.`);
    }
    if (league === "NFL") {
      gaps.push("NFL Week 1 moneyline/spread outputs are preliminary: the 2025 holdout was weak and injury inputs are not complete.");
    }

    const timestamps = [
      ...predictionRows.map((row) => row.asof_ts),
      ...oddsRows.map((row) => row.snapshot_ts),
    ].filter(Boolean).sort();
    return {
      generatedAt: timestamps.at(-1) ?? null,
      predictions: normalized,
      gaps,
    };
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown error";
    return {
      generatedAt: null,
      predictions: [],
      gaps: [...supabaseConfigGaps(), `Supabase ${league} game query failed: ${detail}`],
    };
  }
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
