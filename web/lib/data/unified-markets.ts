import { getCfbMarketFeed } from "@/lib/data/cfb-markets";
import { getMlbResearchBoard, type MlbResearchPrediction } from "@/lib/data/mlb-research";
import { getNflAnytimeTdFeed } from "@/lib/data/nfl-anytime-td";
import { getMlbHomeRunBoardSnapshot } from "@/lib/data/player-markets";
import { getTeamMarketPredictions } from "@/lib/data/team-markets";
import type { Prediction } from "@/lib/data/types";

const CURRENT_MARKET_BUFFER_MS = 5 * 60 * 1000;

export type UnifiedMarketFeed = {
  generatedAt: string | null;
  predictions: Prediction[];
  warnings: string[];
};

export type UnifiedMarketPreparation = {
  predictions: Prediction[];
  missingModelProbability: number;
  hiddenStartedOrInvalid: number;
};

type ResearchSelection = {
  probability: number | null;
  side: "home" | "away" | "over" | "under" | null;
};

function signedLine(value: number | null): string {
  if (value == null) return "";
  return `${value > 0 ? "+" : ""}${value}`;
}

function preferredSide(
  first: number | null | undefined,
  second: number | null | undefined,
  firstSide: ResearchSelection["side"],
  secondSide: ResearchSelection["side"],
): ResearchSelection {
  if (first == null && second == null) return { probability: null, side: null };
  if (second == null || (first != null && first >= second)) {
    return { probability: first ?? null, side: firstSide };
  }
  return { probability: second, side: secondSide };
}

function researchSelection(row: MlbResearchPrediction): ResearchSelection {
  if (row.recommendedProbability != null && row.recommendedSide) {
    const side = row.recommendedSide;
    if (side === "home" || side === "away" || side === "over" || side === "under") {
      return { probability: row.recommendedProbability, side };
    }
  }

  if (row.market === "moneyline") {
    return preferredSide(row.homeWinProb, row.awayWinProb, "home", "away");
  }
  if (row.market === "run_line") {
    return preferredSide(row.pHomeCover15, row.pAwayCoverPlus15, "home", "away");
  }

  const overProbability = row.totalLine === 9.5
    ? row.pOver95
    : row.pOver85 ?? row.pOver95;
  if (overProbability == null) return { probability: null, side: null };
  return overProbability >= 0.5
    ? { probability: overProbability, side: "over" }
    : { probability: 1 - overProbability, side: "under" };
}

function researchPrice(row: MlbResearchPrediction, side: ResearchSelection["side"]): number | null {
  if (row.oddsStatus !== "ok") return null;
  if (row.market === "moneyline") {
    return side === "home" ? row.homePrice ?? null : row.awayPrice ?? null;
  }
  if (row.market === "run_line") {
    return side === "home" ? row.homeRunlinePrice ?? null : row.awayRunlinePrice ?? null;
  }
  return side === "over" ? row.overPrice ?? null : row.underPrice ?? null;
}

function researchLine(row: MlbResearchPrediction, side: ResearchSelection["side"]): number | null {
  if (row.market === "run_line" && row.homeRunlineLine != null) {
    return side === "away" ? -row.homeRunlineLine : row.homeRunlineLine;
  }
  return row.market === "total" ? row.totalLine ?? null : null;
}

function researchSubject(row: MlbResearchPrediction, selection: ResearchSelection): string {
  const side = selection.side;
  if (row.market === "moneyline") {
    const team = side === "home" ? row.homeTeam : row.awayTeam;
    return `${team} moneyline`;
  }
  if (row.market === "run_line") {
    const team = side === "home" ? row.homeTeam : row.awayTeam;
    return `${team} ${signedLine(researchLine(row, side))} run line`;
  }
  return `${row.awayTeam} @ ${row.homeTeam} ${side ?? "total"} ${row.totalLine ?? ""}`.trim();
}

function researchDetailHref(market: MlbResearchPrediction["market"]): string {
  if (market === "run_line") return "/markets/mlb/run-line";
  if (market === "total") return "/markets/mlb/totals";
  return "/markets/mlb";
}

export function mapMlbResearchPrediction(row: MlbResearchPrediction): Prediction {
  const selection = researchSelection(row);
  const modelProbability = selection.probability;
  const edge = row.oddsStatus === "ok" ? row.edge ?? null : null;
  const impliedProbability = modelProbability != null && edge != null
    ? Math.min(1, Math.max(0, modelProbability - edge))
    : null;
  const price = researchPrice(row, selection.side);
  return {
    id: `mlb-research-${row.id}`,
    sport: "MLB",
    league: "MLB",
    gameId: row.gameId,
    eventTime: row.eventTime,
    subject: researchSubject(row, selection),
    homeTeam: row.homeTeam,
    awayTeam: row.awayTeam,
    market: row.market,
    book: price == null ? "model" : row.bestBook ?? "n/a",
    line: researchLine(row, selection.side),
    price,
    modelProbability,
    impliedProbability,
    edge,
    ev: row.oddsStatus === "ok" ? row.ev ?? null : null,
    kelly: row.oddsStatus === "ok" ? row.kelly ?? null : null,
    confidence: null,
    modelVersion: row.modelVersion,
    marketStatus: price == null ? "model_only" : "research",
    detailHref: researchDetailHref(row.market),
    source: "Supabase MLB research market snapshot",
    updatedAt: row.asOfTs,
  };
}

function evDescendingNullLast(left: Prediction, right: Prediction): number {
  if (left.ev == null && right.ev == null) return 0;
  if (left.ev == null) return 1;
  if (right.ev == null) return -1;
  return right.ev - left.ev;
}

export function prepareUnifiedMarketRows(
  rows: Prediction[],
  nowMs = Date.now(),
): UnifiedMarketPreparation {
  const cutoff = nowMs + CURRENT_MARKET_BUFFER_MS;
  const deduplicated = new Map<string, Prediction>();
  let missingModelProbability = 0;
  let hiddenStartedOrInvalid = 0;

  for (const row of rows) {
    if (row.modelProbability == null || !Number.isFinite(row.modelProbability)) {
      missingModelProbability += 1;
      continue;
    }

    const eventTime = row.eventTime ? new Date(row.eventTime).getTime() : Number.NaN;
    if (!Number.isFinite(eventTime) || eventTime <= cutoff) {
      hiddenStartedOrInvalid += 1;
      continue;
    }

    const marketStatus = row.price == null || row.book === "model"
      ? "model_only"
      : row.marketStatus === "supported"
        ? "supported"
        : "research";
    if (!deduplicated.has(row.id)) {
      deduplicated.set(row.id, { ...row, marketStatus });
    }
  }

  return {
    predictions: Array.from(deduplicated.values()).sort(evDescendingNullLast),
    missingModelProbability,
    hiddenStartedOrInvalid,
  };
}

export function deduplicateWarnings(warnings: Array<string | null | undefined>): string[] {
  return Array.from(new Set(warnings.filter((warning): warning is string => Boolean(warning))));
}

export async function getUnifiedMarketFeed(): Promise<UnifiedMarketFeed> {
  const [
    mlbHomeRuns,
    mlbMoneyline,
    mlbRunLine,
    mlbTotal,
    nflTeam,
    nflTd,
    cfb,
    nba,
  ] = await Promise.all([
    getMlbHomeRunBoardSnapshot(),
    getMlbResearchBoard("moneyline"),
    getMlbResearchBoard("run_line"),
    getMlbResearchBoard("total"),
    getTeamMarketPredictions("NFL", { lookaheadDays: 14 }),
    getNflAnytimeTdFeed(),
    getCfbMarketFeed(),
    getTeamMarketPredictions("NBA", { lookaheadDays: 2 }),
  ]);

  const mlbResearch = [mlbMoneyline, mlbRunLine, mlbTotal]
    .flatMap((feed) => feed.predictions)
    .map(mapMlbResearchPrediction);
  const prepared = prepareUnifiedMarketRows([
    ...mlbHomeRuns.rows,
    ...mlbResearch,
    ...nflTeam.predictions,
    ...nflTd.predictions,
    ...cfb.predictions,
    // Tournament outrights are deliberately excluded from the cross-sport board:
    // they run for days, so a start-time cutoff always hides them, and surfacing
    // them would bury the board in model-only rows. See /markets/pga instead.
    ...nba.predictions,
  ]);

  return {
    generatedAt: [
      mlbHomeRuns.completedAt,
      mlbMoneyline.generatedAt,
      mlbRunLine.generatedAt,
      mlbTotal.generatedAt,
      nflTeam.generatedAt,
      nflTd.generatedAt,
      cfb.generatedAt,
      nba.generatedAt,
    ].filter((value): value is string => Boolean(value)).sort().at(-1) ?? null,
    predictions: prepared.predictions,
    warnings: deduplicateWarnings([
      ...mlbHomeRuns.gaps,
      ...mlbMoneyline.gaps,
      ...mlbRunLine.gaps,
      ...mlbTotal.gaps,
      ...nflTeam.gaps,
      ...nflTd.gaps,
      ...cfb.gaps,
      ...nba.gaps,
      prepared.missingModelProbability
        ? `${prepared.missingModelProbability} sportsbook rows are withheld because no model probability is available.`
        : null,
      prepared.hiddenStartedOrInvalid
        ? `${prepared.hiddenStartedOrInvalid} rows are hidden because the event started, begins within five minutes, or has no valid start time.`
        : null,
    ]),
  };
}
