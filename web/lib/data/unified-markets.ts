import { getCfbMarketFeed } from "@/lib/data/cfb-markets";
import { getMlbResearchBoard, type MlbResearchPrediction } from "@/lib/data/mlb-research";
import { getNflAnytimeTdFeed } from "@/lib/data/nfl-anytime-td";
import { getTeamMarketPredictions } from "@/lib/data/team-markets";
import type { Prediction } from "@/lib/data/types";

const CURRENT_MARKET_BUFFER_MS = 5 * 60 * 1000;

function signedLine(value: number | null): string {
  if (value == null) return "";
  return `${value > 0 ? "+" : ""}${value}`;
}

function researchPrice(row: MlbResearchPrediction): number | null {
  if (row.market === "moneyline") {
    return row.recommendedSide === "home" ? row.homePrice ?? null : row.awayPrice ?? null;
  }
  if (row.market === "run_line") {
    return row.recommendedSide === "home" ? row.homeRunlinePrice ?? null : row.awayRunlinePrice ?? null;
  }
  return row.recommendedSide === "over" ? row.overPrice ?? null : row.underPrice ?? null;
}

function researchLine(row: MlbResearchPrediction): number | null {
  if (row.market === "run_line" && row.homeRunlineLine != null) {
    return row.recommendedSide === "away" ? -row.homeRunlineLine : row.homeRunlineLine;
  }
  return row.market === "total" ? row.totalLine ?? null : null;
}

function researchSubject(row: MlbResearchPrediction): string {
  const side = row.recommendedSide;
  if (row.market === "moneyline") {
    const team = side === "home" ? row.homeTeam : row.awayTeam;
    return `${team} moneyline`;
  }
  if (row.market === "run_line") {
    const team = side === "home" ? row.homeTeam : row.awayTeam;
    return `${team} ${signedLine(researchLine(row))} run line`;
  }
  return `${row.awayTeam} @ ${row.homeTeam} ${side ?? "total"} ${row.totalLine ?? ""}`.trim();
}

export function mapMlbResearchPrediction(row: MlbResearchPrediction): Prediction {
  const modelProbability = row.recommendedProbability ?? null;
  const edge = row.edge ?? null;
  const impliedProbability = modelProbability != null && edge != null
    ? Math.min(1, Math.max(0, modelProbability - edge))
    : null;
  return {
    id: `mlb-research-${row.id}`,
    sport: "MLB",
    league: "MLB",
    gameId: row.gameId,
    eventTime: row.eventTime,
    subject: researchSubject(row),
    homeTeam: row.homeTeam,
    awayTeam: row.awayTeam,
    market: row.market,
    book: row.bestBook ?? "n/a",
    line: researchLine(row),
    price: researchPrice(row),
    modelProbability,
    impliedProbability,
    edge,
    ev: row.ev ?? null,
    kelly: row.kelly ?? null,
    confidence: null,
    modelVersion: row.modelVersion,
    source: "Supabase MLB research market snapshot",
    updatedAt: row.asOfTs,
  };
}

export function filterCurrentPositiveEv(
  predictions: Prediction[],
  nowMs = Date.now(),
): Prediction[] {
  const cutoff = nowMs + CURRENT_MARKET_BUFFER_MS;
  return predictions
    .filter((prediction) => {
      const eventTime = prediction.eventTime ? new Date(prediction.eventTime).getTime() : Number.NaN;
      return Number.isFinite(eventTime)
        && eventTime > cutoff
        && prediction.ev != null
        && Number.isFinite(prediction.ev)
        && prediction.ev > 0;
    })
    .sort((left, right) => (right.ev ?? Number.NEGATIVE_INFINITY) - (left.ev ?? Number.NEGATIVE_INFINITY));
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

export async function getUnifiedResearchMarketFeed(): Promise<{
  predictions: Prediction[];
  gaps: string[];
}> {
  const [mlbMoneyline, mlbRunLine, mlbTotal, nflTeam, nflTd, cfb, nba] = await Promise.all([
    getMlbResearchBoard("moneyline"),
    getMlbResearchBoard("run_line"),
    getMlbResearchBoard("total"),
    getTeamMarketPredictions("NFL", { lookaheadDays: 14 }),
    getNflAnytimeTdFeed(),
    getCfbMarketFeed(),
    getTeamMarketPredictions("NBA", { lookaheadDays: 2 }),
  ]);
  const mlbRows = [mlbMoneyline, mlbRunLine, mlbTotal]
    .flatMap((feed) => feed.predictions)
    .filter((row) => row.oddsStatus === "ok")
    .map(mapMlbResearchPrediction);
  const predictions = filterCurrentPositiveEv([
    ...mlbRows,
    ...nflTeam.predictions,
    ...nflTd.predictions,
    ...cfb.predictions,
    ...nba.predictions,
  ]);
  return {
    predictions,
    gaps: unique([
      "Research EV is model-implied, not a validated bankroll recommendation. Use each market's guardrails and evidence notes.",
      ...mlbMoneyline.gaps,
      ...mlbRunLine.gaps,
      ...mlbTotal.gaps,
      ...nflTeam.gaps,
      ...nflTd.gaps,
      ...cfb.gaps,
      ...nba.gaps,
      "NBA rows join this table automatically when the in-season daily pipeline has current predictions and prices.",
    ]),
  };
}
