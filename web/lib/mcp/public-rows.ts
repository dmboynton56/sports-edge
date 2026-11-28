import { formatFairAmericanOdds } from "@/lib/fair-odds";
import type { GameExplanation } from "@/lib/data/explanations";
import type { MlbResearchPrediction } from "@/lib/data/mlb-research";
import type { GameResultRow } from "@/lib/data/results";
import type { TeamSlateGame } from "@/lib/data/team-markets";
import type { Prediction } from "@/lib/data/types";
import type { MobileGameExplanation, MobileMarket } from "@/lib/mobile/types";
import { publicationStatus } from "@/lib/mcp/envelope";
import {
  MCP_MAX_FEATURES,
  type McpEdgeRow,
  type McpExplanationFeature,
  type McpGameDetail,
  type McpPlayerMarket,
  type McpPublicationStatus,
  type McpResultGame,
  type McpSlateGame,
} from "@/lib/mcp/types";

export function rankEdgeRows(rows: McpEdgeRow[], sort: "point_edge" | "ev", limit: number): McpEdgeRow[] {
  const metric = sort === "ev"
    ? (row: McpEdgeRow) => row.ev
    : (row: McpEdgeRow) => row.pointEdge;
  return [...rows]
    .filter((row) => metric(row) != null)
    .sort((left, right) => Math.abs(metric(right) ?? 0) - Math.abs(metric(left) ?? 0))
    .slice(0, limit);
}

export function edgeRowFromSlate(game: TeamSlateGame, status: McpPublicationStatus): McpEdgeRow {
  return {
    id: `${game.league}-${game.gameId}`,
    league: game.league,
    title: `${game.awayTeam} @ ${game.homeTeam}`,
    homeTeam: game.homeTeam,
    awayTeam: game.awayTeam,
    market: "spread",
    line: game.bookSpread,
    modelProbability: game.homeWinProb,
    pointEdge: game.edgePts,
    ev: null,
    modelVersion: game.modelVersion,
    status,
  };
}

export function edgeRowFromPrediction(prediction: Prediction, status: McpPublicationStatus): McpEdgeRow {
  return {
    id: prediction.id,
    league: prediction.league,
    title: prediction.subject,
    homeTeam: prediction.homeTeam ?? null,
    awayTeam: prediction.awayTeam ?? null,
    market: prediction.market,
    line: prediction.line,
    modelProbability: prediction.modelProbability,
    pointEdge: null,
    ev: prediction.ev,
    modelVersion: prediction.modelVersion,
    status: publicationStatus({
      marketStatus: prediction.marketStatus,
      registryStatus: status,
    }),
  };
}

export function edgeRowFromMlbResearch(row: MlbResearchPrediction): McpEdgeRow {
  const title = `${row.awayTeam} @ ${row.homeTeam} ${row.market}`;
  return {
    id: row.id,
    league: "MLB",
    title,
    homeTeam: row.homeTeam,
    awayTeam: row.awayTeam,
    market: row.market,
    line: row.homeRunlineLine ?? row.totalLine ?? null,
    modelProbability: row.recommendedProbability ?? row.homeWinProb ?? row.pHomeCover15 ?? row.pOver95 ?? row.pOver85 ?? null,
    pointEdge: row.edge ?? null,
    ev: row.oddsStatus === "ok" ? row.ev ?? null : null,
    modelVersion: row.modelVersion,
    status: "research",
  };
}

export function slateGameFromTeam(game: TeamSlateGame): McpSlateGame {
  return {
    gameId: game.gameId,
    league: game.league,
    title: `${game.awayTeam} @ ${game.homeTeam}`,
    homeTeam: game.homeTeam,
    awayTeam: game.awayTeam,
    gameDate: game.gameDate,
    eventTime: game.gameTimeUtc,
    bookSpread: game.bookSpread,
    modelSpread: game.modelSpread,
    pointEdge: game.edgePts,
    homeWinProb: game.homeWinProb,
    modelVersion: game.modelVersion,
    freshnessStatus: game.freshnessStatus,
  };
}

export function matchSlateGame(games: TeamSlateGame[], id?: string, team?: string): TeamSlateGame[] {
  const needle = (id ?? team ?? "").trim();
  if (!needle) return [];
  const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(needle);
  if (uuid) return games.filter((game) => game.gameId === needle);
  const token = needle.toUpperCase();
  return games.filter((game) => {
    const home = game.homeTeam.toUpperCase();
    const away = game.awayTeam.toUpperCase();
    const title = `${away} @ ${home}`;
    return home === token || away === token || title.includes(token);
  });
}

export function topExplanationFeatures(
  explanation: GameExplanation | MobileGameExplanation | null,
): McpExplanationFeature[] {
  const features = explanation?.topFeatures ?? [];
  return [...features]
    .map((feature) => ({
      feature: feature.feature,
      impact: feature.impact,
      value: Number.isFinite(feature.value) ? feature.value : null,
    }))
    .sort((left, right) => Math.abs(right.impact) - Math.abs(left.impact))
    .slice(0, MCP_MAX_FEATURES);
}

export function gameDetailFromMobile(
  market: MobileMarket,
  explanation: MobileGameExplanation | null,
): McpGameDetail {
  return {
    gameId: market.gameId,
    league: market.league,
    title: market.title,
    homeTeam: market.homeTeam,
    awayTeam: market.awayTeam,
    line: market.line,
    modelSpread: null,
    homeWinProb: market.modelProbability,
    pointEdge: market.edge,
    modelVersion: market.modelVersion,
    injuryAdjusted: Boolean(market.injuryAdjusted || explanation?.injuryAdjusted),
    injuryDataMissing: Boolean(market.injuryDataMissing),
    topFeatures: topExplanationFeatures(explanation),
  };
}

export function publicPlayerMarket(prediction: Prediction): McpPlayerMarket {
  return {
    id: prediction.id,
    league: prediction.league,
    player: prediction.player ?? prediction.subject,
    team: prediction.homeTeam ?? null,
    opponent: prediction.awayTeam ?? null,
    market: prediction.market,
    modelProbability: prediction.modelProbability,
    fairOdds: formatFairAmericanOdds(prediction.modelProbability),
    modelVersion: prediction.modelVersion,
    asOf: prediction.updatedAt ?? null,
  };
}

export function resultGameFromRow(row: GameResultRow): McpResultGame {
  return {
    gameDate: row.game_date,
    homeTeam: row.home_team,
    awayTeam: row.away_team,
    homeScore: row.home_score,
    awayScore: row.away_score,
    bookSpread: row.book_spread,
    modelSpread: row.my_spread,
    spreadResult: row.spread_result,
    winnerResult: row.winner_result,
    units: row.flat_ats_units,
    modelVersion: row.model_version,
  };
}
