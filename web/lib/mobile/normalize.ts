import type { Prediction } from "@/lib/data/types";
import { isFiniteNumber } from "@/lib/data/json";
import { getGameExplanation, type GameExplanation } from "@/lib/data/explanations";
import { getEvaluationsBundle } from "@/lib/data/evaluations";
import { getMlbHomeRunFeed } from "@/lib/data/player-markets";
import { getPgaBoardData } from "@/lib/data/player-markets";
import { getPerformanceHistory } from "@/lib/data/performance";
import { getTeamSlateFeed, getTeamSlateGame, type TeamSlateFeed, type TeamSlateGame } from "@/lib/data/team-markets";
import type { MobileEnvelope, MobileFreshness, MobileFreshnessStatus, MobileGameDetailData, MobileHomeData, MobileInsightsData, MobileLeague, MobileMarket, MobileMarketsData, MobilePerformanceData, MobilePerformanceRecord, MobileSource } from "@/lib/mobile/types";
import { MOBILE_SCHEMA_VERSION } from "@/lib/mobile/types";

const MOBILE_LEAGUES = { NBA: true, NFL: true, MLB: true, PGA: true } as const;

function isMobileLeague(value: string): value is MobileLeague {
  return Object.keys(MOBILE_LEAGUES).includes(value);
}

function toMobileLeague(value: string, fallback: MobileLeague): MobileLeague {
  return isMobileLeague(value) ? value : fallback;
}

function unique(values: string[]) {
  return Array.from(new Set(values.filter(Boolean)));
}

function ageSeconds(updatedAt: string | null) {
  if (!updatedAt) return null;
  const timestamp = Date.parse(updatedAt);
  if (!Number.isFinite(timestamp)) return null;
  return Math.max(0, Math.trunc((Date.now() - timestamp) / 1000));
}

export function freshnessFor(
  updatedAt: string | null,
  source: MobileSource,
  gaps: string[] = [],
): MobileFreshness {
  const age = ageSeconds(updatedAt);
  let status: MobileFreshnessStatus = "missing";
  if (source === "unavailable" || !updatedAt || age == null) status = "missing";
  else if (source === "fixture") status = "fresh";
  else if (gaps.some((gap) => /offline|cache/i.test(gap))) status = "offline";
  else if (age > 24 * 60 * 60) status = "stale";
  else status = "fresh";
  return { status, source, updatedAt, ageSeconds: age };
}

export function envelope<T>(
  data: T,
  gaps: string[],
  updatedAt: string | null,
  source: MobileSource,
): MobileEnvelope<T> {
  return {
    schemaVersion: MOBILE_SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    data,
    gaps: unique(gaps),
    freshness: freshnessFor(updatedAt, source, gaps),
  };
}

function teamMarket(game: TeamSlateGame): MobileMarket {
  return {
    id: `${game.league}-${game.gameId}`,
    gameId: game.gameId,
    league: toMobileLeague(game.league.toUpperCase(), "NBA"),
    kind: "team_spread",
    title: `${game.awayTeam} @ ${game.homeTeam}`,
    subtitle: game.week != null ? `Week ${game.week}` : game.gameDate ?? "Upcoming game",
    eventTime: game.gameTimeUtc,
    homeTeam: game.homeTeam,
    awayTeam: game.awayTeam,
    subject: null,
    market: "spread",
    book: "market",
    line: game.bookSpread,
    price: null,
    modelProbability: game.homeWinProb,
    impliedProbability: null,
    edge: game.edgePts,
    ev: null,
    confidence: game.homeWinProb,
    modelVersion: game.modelVersion,
    freshnessStatus: game.freshnessStatus,
    predictionTs: game.predictionTs,
    oddsTs: game.oddsTs,
    injuryAdjusted: Boolean(game.injuryAdjusted),
    injuryDataMissing: Boolean(game.injuryDataMissing),
  };
}

function playerMarket(prediction: Prediction): MobileMarket {
  return {
    id: prediction.id,
    gameId: prediction.gameId,
    league: toMobileLeague(prediction.league.toUpperCase(), "MLB"),
    kind: "player_market",
    title: prediction.subject,
    subtitle: prediction.player ?? prediction.market,
    eventTime: prediction.eventTime,
    homeTeam: prediction.homeTeam ?? null,
    awayTeam: prediction.awayTeam ?? null,
    subject: prediction.player ?? prediction.subject,
    market: prediction.market,
    book: prediction.book,
    line: prediction.line,
    price: prediction.price,
    modelProbability: prediction.modelProbability,
    impliedProbability: prediction.impliedProbability,
    edge: prediction.edge,
    ev: prediction.ev,
    confidence: prediction.confidence,
    modelVersion: prediction.modelVersion,
    freshnessStatus: prediction.updatedAt ? "fresh" : "missing",
    predictionTs: prediction.updatedAt ?? null,
    oddsTs: prediction.updatedAt ?? null,
    injuryAdjusted: false,
    injuryDataMissing: false,
  };
}

function sortByEdge(markets: MobileMarket[]) {
  return [...markets].sort((left, right) => Math.abs(right.edge ?? 0) - Math.abs(left.edge ?? 0));
}

function sourceForFeed(feed: TeamSlateFeed): MobileSource {
  return feed.games.length ? "supabase" : "unavailable";
}

function latestTeamTimestamp(games: TeamSlateGame[]) {
  return games
    .map((game) => game.predictionTs ?? game.oddsTs)
    .filter((timestamp): timestamp is string => Boolean(timestamp))
    .sort()
    .at(-1) ?? null;
}

async function teamMarkets(league: "NBA" | "NFL") {
  const feed = await getTeamSlateFeed(league, { lookaheadDays: league === "NFL" ? 7 : 1 });
  const markets = feed.games.map(teamMarket);
  return {
    data: {
      league,
      windowStart: feed.windowStart,
      windowEnd: feed.windowEnd,
      markets,
    } satisfies MobileMarketsData,
    gaps: feed.gaps,
    updatedAt: latestTeamTimestamp(feed.games),
    source: sourceForFeed(feed),
  };
}

async function playerMarkets(league: "MLB" | "PGA") {
  if (league === "MLB") {
    const feed = await getMlbHomeRunFeed();
    const markets = feed.predictions.map(playerMarket);
    return {
      data: { league, windowStart: null, windowEnd: null, markets } satisfies MobileMarketsData,
      gaps: feed.gaps,
      updatedAt: feed.generatedAt,
      source: feed.dataSource === "static_json" ? "static_json" : markets.length ? "supabase" : "unavailable",
    } as const;
  }
  const board = await getPgaBoardData();
  const markets = (board.normalizedMarkets ?? []).map(playerMarket);
  return {
    data: { league, windowStart: null, windowEnd: null, markets } satisfies MobileMarketsData,
    gaps: board.gaps ?? [],
    updatedAt: board.generatedAt ?? null,
    source: board.dataSource === "static_json" ? "static_json" : markets.length ? "supabase" : "unavailable",
  } as const;
}

export async function getMobileMarkets(league: MobileLeague) {
  return league === "NBA" || league === "NFL" ? teamMarkets(league) : playerMarkets(league);
}

export async function getMobileHome() {
  const [nba, nfl] = await Promise.all([teamMarkets("NBA"), teamMarkets("NFL")]);
  const allMarkets = [...nba.data.markets, ...nfl.data.markets];
  const topEdges = sortByEdge(allMarkets).slice(0, 8);
  const data: MobileHomeData = {
    topEdges,
    leagueSummaries: [nba, nfl].map((feed) => ({
      league: feed.data.league,
      marketCount: feed.data.markets.length,
      topEdge: sortByEdge(feed.data.markets)[0]?.edge ?? null,
    })),
  };
  const gaps = unique([...nba.gaps, ...nfl.gaps]);
  const timestamps = [nba.updatedAt, nfl.updatedAt].filter((timestamp): timestamp is string => Boolean(timestamp));
  const updatedAt = timestamps.sort().at(-1) ?? null;
  const source: MobileSource = allMarkets.length ? "supabase" : "unavailable";
  return { data, gaps, updatedAt, source };
}

export async function getMobileGameDetail(league: "NBA" | "NFL", gameId: string) {
  const [game, explanation] = await Promise.all([
    getTeamSlateGame(league, gameId),
    getGameExplanation(gameId, league),
  ]);
  const data: MobileGameDetailData | null = game
    ? { game: teamMarket(game), explanation: explanation ? mapExplanation(explanation) : null }
    : null;
  return {
    data,
    gaps: data ? [] : [`No ${league} game found for ${gameId}.`],
    updatedAt: game?.predictionTs ?? game?.oddsTs ?? null,
    source: data ? "supabase" : "unavailable",
  } as const;
}

function mapExplanation(explanation: GameExplanation) {
  return {
    gameId: explanation.gameId,
    league: toMobileLeague(explanation.league.toUpperCase(), "NBA"),
    modelVersion: explanation.modelVersion,
    predictionTs: explanation.predictionTs,
    topFeatures: explanation.topFeatures,
    injuryAdjusted: explanation.injuryAdjusted,
    homeInjuryDelta: explanation.homeInjuryDelta,
    awayInjuryDelta: explanation.awayInjuryDelta,
    baseVsAdjusted: explanation.baseVsAdjusted,
  };
}

function performanceRecord(record: Awaited<ReturnType<typeof getPerformanceHistory>>["records"][number]): MobilePerformanceRecord {
  return {
    league: record.sport,
    modelVersion: record.modelVersion,
    season: record.season,
    market: record.market,
    sampleSize: record.sampleSize,
    roi: record.roi,
    units: record.units,
    bets: record.bets,
    wins: record.wins,
    losses: record.losses,
    pushes: record.pushes,
    productionStatus: record.productionStatus,
    gates: record.productionGates,
  };
}

export async function getMobilePerformance() {
  const history = await getPerformanceHistory();
  const data: MobilePerformanceData = {
    generatedAt: history.generatedAt,
    records: history.records.map(performanceRecord),
  };
  const source: MobileSource = "static_json";
  return {
    data,
    gaps: history.gaps,
    updatedAt: history.generatedAt,
    source,
  };
}

export async function getMobileInsights() {
  const [bundle, history] = await Promise.all([getEvaluationsBundle(), getPerformanceHistory()]);
  const data: MobileInsightsData = {
    dataQuality: [
      {
        id: "serving",
        label: "Serving feed",
        status: bundle.gaps.length ? "warning" : "ok",
        updatedAt: history.generatedAt,
        detail: bundle.gaps.length ? bundle.gaps.join(" ") : "Mobile API and public serving tables are reachable.",
      },
      {
        id: "freshness",
        label: "Performance artifact",
        status: history.generatedAt ? "ok" : "missing",
        updatedAt: history.generatedAt,
        detail: history.generatedAt ? "Performance history has a generated timestamp." : "No performance artifact timestamp is available.",
      },
      {
        id: "gaps",
        label: "Known gaps",
        status: history.gaps.length ? "warning" : "ok",
        updatedAt: history.generatedAt,
        detail: history.gaps.length ? `${history.gaps.length} data-quality notes need attention.` : "No known performance data gaps.",
      },
    ],
    evaluations: bundle.evaluations.map((evaluation) => ({
      id: evaluation.id,
      league: evaluation.league,
      modelVersion: evaluation.modelVersion,
      evaluationName: evaluation.evaluationName,
      generatedAt: evaluation.generatedAt,
      status: evaluation.status,
      roi: numberMetric(evaluation.metrics, ["supabase_ats_roi", "flat_roi", "bigquery_default_roi", "best_reported_sweep_roi"]),
      auc: numberMetric(evaluation.metrics, ["auc", "roc_auc", "bigquery_auc", "win_auc"]),
    })),
    strategies: bundle.strategies.map((strategy) => ({
      id: strategy.id,
      league: strategy.league,
      modelVersion: strategy.modelVersion,
      strategyId: strategy.strategyId,
      market: strategy.market,
      sampleSize: strategy.sampleSize,
      bets: strategy.bets,
      roi: strategy.roi,
    })),
  };
  return {
    data,
    gaps: unique([...bundle.gaps, ...history.gaps]),
    updatedAt: history.generatedAt,
    source: bundle.gaps.length ? "mixed" : "supabase",
  } as const;
}

type MetricMap = Record<string, string | number | null | undefined>;

function numberMetric(metrics: MetricMap, names: string[]) {
  for (const name of names) {
    const value = metrics[name];
    if (isFiniteNumber(value)) return value;
  }
  return null;
}
