import { getEvaluationsBundle } from "@/lib/data/evaluations";
import { getMlbResearchBoard } from "@/lib/data/mlb-research";
import { getNflAnytimeTdFeed } from "@/lib/data/nfl-anytime-td";
import { getMlbHomeRunFeed, getPgaBoardData } from "@/lib/data/player-markets";
import {
  getGameResultRows,
  getMlbHomeRunResultRows,
  getPgaResultRows,
  summarizeGameResults,
  summarizeMlbHr,
  summarizePga,
} from "@/lib/data/results";
import {
  getTeamMarketPredictions,
  getTeamSlateFeed,
  getTeamSlateGame,
  PREFERRED_MODEL_VERSION,
  type TeamSlateFeed,
} from "@/lib/data/team-markets";
import { isJsonString } from "@/lib/data/json";
import { getMobileGameDetail, getMobilePerformance } from "@/lib/mobile/normalize";
import { mcpMeta, publicationStatus, latestTimestamp, uniqueGaps } from "@/lib/mcp/envelope";
import {
  edgeRowFromMlbResearch,
  edgeRowFromPrediction,
  edgeRowFromSlate,
  gameDetailFromMobile,
  matchSlateGame,
  publicPlayerMarket,
  rankEdgeRows,
  resultGameFromRow,
  slateGameFromTeam,
} from "@/lib/mcp/public-rows";
import {
  MCP_MAX_PLAYER_ROWS,
  MCP_MAX_RESULT_ROWS,
  MCP_MAX_SLATE_ROWS,
  type GetGameInput,
  type GetGameResult,
  type GetModelPerformanceInput,
  type GetModelPerformanceResult,
  type GetPlayerMarketsInput,
  type GetPlayerMarketsResult,
  type GetResultsInput,
  type GetResultsResult,
  type GetSlateInput,
  type GetSlateResult,
  type GetStatusResult,
  type ListEdgesInput,
  type ListEdgesResult,
  type McpEdgeRow,
  type McpLeagueStatus,
  type McpPublicationStatus,
  type McpResultsSummary,
} from "@/lib/mcp/types";
import { clampEdgeLimit, clampLookaheadDays, filterMcpResults } from "@/lib/mcp/windows";

export type McpDeps = {
  getTeamSlateFeed: typeof getTeamSlateFeed;
  getTeamSlateGame: typeof getTeamSlateGame;
  getTeamMarketPredictions: typeof getTeamMarketPredictions;
  getMobileGameDetail: typeof getMobileGameDetail;
  getMobilePerformance: typeof getMobilePerformance;
  getMlbResearchBoard: typeof getMlbResearchBoard;
  getMlbHomeRunFeed: typeof getMlbHomeRunFeed;
  getPgaBoardData: typeof getPgaBoardData;
  getNflAnytimeTdFeed: typeof getNflAnytimeTdFeed;
  getGameResultRows: typeof getGameResultRows;
  getMlbHomeRunResultRows: typeof getMlbHomeRunResultRows;
  getPgaResultRows: typeof getPgaResultRows;
  getEvaluationsBundle: typeof getEvaluationsBundle;
};

export const liveMcpDeps: McpDeps = {
  getTeamSlateFeed,
  getTeamSlateGame,
  getTeamMarketPredictions,
  getMobileGameDetail,
  getMobilePerformance,
  getMlbResearchBoard,
  getMlbHomeRunFeed,
  getPgaBoardData,
  getNflAnytimeTdFeed,
  getGameResultRows,
  getMlbHomeRunResultRows,
  getPgaResultRows,
  getEvaluationsBundle,
};

function performanceStatus(
  records: Awaited<ReturnType<typeof getMobilePerformance>>["data"]["records"],
  league: string,
): McpPublicationStatus {
  const record = records.find((item) => item.league.toUpperCase() === league);
  return publicationStatus({ productionStatus: record?.productionStatus ?? null });
}

function firstVersion(rows: { modelVersion?: string | null }[]): string | null {
  const versions = Array.from(new Set(rows.map((row) => row.modelVersion).filter((value): value is string => Boolean(value))));
  if (!versions.length) return null;
  return versions.length === 1 ? versions[0] : versions.join(", ");
}

async function teamSlate(league: "NBA" | "NFL", lookaheadDays: number | undefined, deps: McpDeps) {
  return deps.getTeamSlateFeed(league, { lookaheadDays: clampLookaheadDays(league, lookaheadDays) });
}

export async function handleGetStatus(deps: McpDeps = liveMcpDeps): Promise<GetStatusResult> {
  const [nba, nfl, mlbMl, mlbHr, pga, performance] = await Promise.all([
    teamSlate("NBA", undefined, deps),
    teamSlate("NFL", undefined, deps),
    deps.getMlbResearchBoard("moneyline"),
    deps.getMlbHomeRunFeed(),
    deps.getPgaBoardData(),
    deps.getMobilePerformance(),
  ]);

  const nbaStatus = performanceStatus(performance.data.records, "NBA");
  const nflStatus = performanceStatus(performance.data.records, "NFL");
  const leagues: McpLeagueStatus[] = [
    leagueStatus("NBA", nba, nbaStatus, nba.games.map((game) => game.modelVersion), nba.games.map((game) => game.predictionTs)),
    leagueStatus("NFL", nfl, nflStatus, nfl.games.map((game) => game.modelVersion), nfl.games.map((game) => game.predictionTs)),
    {
      league: "MLB",
      as_of: latestTimestamp([mlbMl.generatedAt, mlbHr.generatedAt]),
      model_version: mlbMl.predictions[0]?.modelVersion ?? mlbHr.modelVersion,
      freshness: mcpMeta(
        latestTimestamp([mlbMl.generatedAt, mlbHr.generatedAt]),
        mlbMl.dataSource === "unavailable" && mlbHr.dataSource === "unavailable" ? "unavailable" : "supabase",
        uniqueGaps([...mlbMl.gaps, ...mlbHr.gaps]),
        mlbMl.predictions[0]?.modelVersion ?? mlbHr.modelVersion,
        "research",
      ).freshness,
      status: "research",
      last_publish: latestTimestamp([mlbMl.generatedAt, mlbHr.generatedAt]),
      rowCount: mlbMl.predictions.length + mlbHr.predictions.length,
      gaps: uniqueGaps([...mlbMl.gaps, ...mlbHr.gaps]),
    },
    {
      league: "PGA",
      as_of: pga.generatedAt ?? null,
      model_version: isJsonString(pga.predictionMeta?.model_version) ? pga.predictionMeta.model_version : null,
      freshness: mcpMeta(
        pga.generatedAt ?? null,
        pga.dataSource === "supabase_predictions" ? "supabase" : pga.dataSource === "static_json" ? "static_json" : "unavailable",
        pga.gaps ?? [],
        isJsonString(pga.predictionMeta?.model_version) ? pga.predictionMeta.model_version : null,
        "model-only",
      ).freshness,
      status: "model-only",
      last_publish: pga.generatedAt ?? null,
      rowCount: pga.normalizedMarkets?.length ?? pga.predictions?.length ?? 0,
      gaps: uniqueGaps(pga.gaps ?? []),
    },
  ];

  const gaps = uniqueGaps(leagues.flatMap((league) => league.gaps));
  const asOf = latestTimestamp(leagues.map((league) => league.as_of));
  const stale = leagues.some((league) => league.freshness.status === "stale" || league.freshness.status === "missing");
  if (stale) gaps.push("One or more league feeds are stale or missing. Do not quote them as current.");
  const missingSupabase = gaps.some((gap) => gap.includes("missing NEXT_PUBLIC_SUPABASE") || gap.includes("missing SUPABASE"));
  const source = missingSupabase ? (asOf ? "mixed" : "unavailable") : "supabase";

  return {
    ...mcpMeta(asOf, source, gaps, firstVersion(leagues.map((league) => ({ modelVersion: league.model_version }))), "research"),
    leagues,
  };
}

function leagueStatus(
  league: "NBA" | "NFL",
  feed: TeamSlateFeed,
  status: McpPublicationStatus,
  versions: (string | null)[],
  timestamps: (string | null)[],
): McpLeagueStatus {
  const asOf = latestTimestamp(timestamps);
  const modelVersion = firstVersion(versions.map((modelVersion) => ({ modelVersion }))) ?? PREFERRED_MODEL_VERSION[league];
  const source = feed.games.length ? "supabase" as const : "unavailable" as const;
  const gaps = uniqueGaps([
    ...feed.gaps,
    feed.games.length === 0 ? `No ${league} games in window ${feed.windowStart} to ${feed.windowEnd}.` : null,
  ]);
  return {
    league,
    as_of: asOf,
    model_version: modelVersion,
    freshness: mcpMeta(asOf, source, gaps, modelVersion, status).freshness,
    status,
    last_publish: asOf,
    rowCount: feed.games.length,
    gaps,
  };
}

export async function handleListEdges(input: ListEdgesInput, deps: McpDeps = liveMcpDeps): Promise<ListEdgesResult> {
  const sort = input.sort;
  const limit = clampEdgeLimit(input.limit);
  const league = input.league;
  const performance = await deps.getMobilePerformance();
  const rows: McpEdgeRow[] = [];
  const gaps: string[] = [];
  const timestamps: (string | null)[] = [];
  const versions: { modelVersion?: string | null }[] = [];

  const loadTeamPoint = async (sport: "NBA" | "NFL") => {
    const feed = await teamSlate(sport, undefined, deps);
    const status = performanceStatus(performance.data.records, sport);
    gaps.push(...feed.gaps);
    timestamps.push(...feed.games.map((game) => game.predictionTs));
    versions.push(...feed.games.map((game) => ({ modelVersion: game.modelVersion })));
    rows.push(...feed.games.map((game) => edgeRowFromSlate(game, status)));
  };

  const loadTeamEv = async (sport: "NBA" | "NFL") => {
    const feed = await deps.getTeamMarketPredictions(sport);
    const status = performanceStatus(performance.data.records, sport);
    gaps.push(...feed.gaps);
    timestamps.push(feed.generatedAt);
    versions.push(...feed.predictions.map((row) => ({ modelVersion: row.modelVersion })));
    rows.push(...feed.predictions.map((row) => edgeRowFromPrediction(row, status)));
  };

  const loadMlb = async () => {
    const boards = await Promise.all([
      deps.getMlbResearchBoard("moneyline"),
      deps.getMlbResearchBoard("run_line"),
      deps.getMlbResearchBoard("total"),
    ]);
    for (const board of boards) {
      gaps.push(...board.gaps);
      timestamps.push(board.generatedAt);
      versions.push(...board.predictions.map((row) => ({ modelVersion: row.modelVersion })));
      rows.push(...board.predictions.map(edgeRowFromMlbResearch));
    }
    gaps.push("MLB research edges are probability edges versus captured books, not NBA/NFL spread point differentials.");
  };

  if (sort === "ev") {
    if (!league || league === "NBA") await loadTeamEv("NBA");
    if (!league || league === "NFL") await loadTeamEv("NFL");
    if (league === "MLB") await loadMlb();
    if (league === "PGA") gaps.push("PGA player markets are model probabilities only; no sportsbook EV is published.");
  } else {
    if (!league || league === "NBA") await loadTeamPoint("NBA");
    if (!league || league === "NFL") await loadTeamPoint("NFL");
    if (league === "MLB") await loadMlb();
    if (league === "PGA") gaps.push("PGA has no point-edge ranking; use get_player_markets or get_slate.");
  }

  const ranked = rankEdgeRows(rows, sort, limit);
  if (sort === "ev" && !ranked.length) {
    gaps.push("No priced EV rows are available. Missing books fail closed; EV is not invented from model fair odds.");
  }
  if (sort === "point_edge" && !ranked.length) {
    gaps.push("No point-edge rows are available for this league and window.");
  }

  const asOf = latestTimestamp(timestamps);
  const status = ranked[0]?.status ?? (league === "MLB" ? "research" : "candidate");
  return {
    ...mcpMeta(asOf, asOf ? "supabase" : "unavailable", uniqueGaps(gaps), firstVersion(versions) ?? ranked[0]?.modelVersion ?? null, status),
    sort,
    league: league ?? "ALL",
    limit,
    rows: ranked,
  };
}

export async function handleGetSlate(input: GetSlateInput, deps: McpDeps = liveMcpDeps): Promise<GetSlateResult> {
  const league = input.league;
  const lookaheadDays = clampLookaheadDays(league, input.lookaheadDays);
  if (league === "NBA" || league === "NFL") {
    const feed = await teamSlate(league, lookaheadDays, deps);
    const performance = await deps.getMobilePerformance();
    const status = performanceStatus(performance.data.records, league);
    const games = feed.games.slice(0, MCP_MAX_SLATE_ROWS).map(slateGameFromTeam);
    const asOf = latestTimestamp(feed.games.map((game) => game.predictionTs ?? game.oddsTs));
    const gaps = uniqueGaps([
      ...feed.gaps,
      games.length === 0 ? `Empty ${league} slate for ${feed.windowStart} to ${feed.windowEnd}.` : null,
      `Date window capped at ${lookaheadDays} day(s) (${feed.windowStart} to ${feed.windowEnd}).`,
    ]);
    return {
      ...mcpMeta(asOf, games.length ? "supabase" : "unavailable", gaps, firstVersion(games) ?? PREFERRED_MODEL_VERSION[league], status),
      league,
      windowStart: feed.windowStart,
      windowEnd: feed.windowEnd,
      games,
      event: null,
    };
  }

  if (league === "MLB") {
    const board = await deps.getMlbResearchBoard("moneyline");
    const games = board.predictions.slice(0, MCP_MAX_SLATE_ROWS).map((row) => ({
      gameId: row.gameId,
      league: "MLB",
      title: `${row.awayTeam} @ ${row.homeTeam}`,
      homeTeam: row.homeTeam,
      awayTeam: row.awayTeam,
      gameDate: row.gameDate,
      eventTime: row.eventTime,
      bookSpread: null,
      modelSpread: null,
      pointEdge: row.edge ?? null,
      homeWinProb: row.homeWinProb ?? null,
      modelVersion: row.modelVersion,
      freshnessStatus: row.oddsStatus,
    }));
    const gaps = uniqueGaps([
      ...board.gaps,
      games.length === 0 ? `Empty MLB research slate for ${board.slateDate}.` : null,
      "MLB game research is PropLine-first and labeled research, not production.",
    ]);
    return {
      ...mcpMeta(board.generatedAt, board.dataSource === "supabase" ? "supabase" : "unavailable", gaps, games[0]?.modelVersion ?? null, "research"),
      league,
      windowStart: board.slateDate,
      windowEnd: board.slateDate,
      games,
      event: null,
    };
  }

  const board = await deps.getPgaBoardData();
  const players = (board.normalizedMarkets ?? []).filter((row) => row.market === "win").slice(0, MCP_MAX_SLATE_ROWS);
  const eventName = isJsonString(board.event?.name) ? board.event.name : null;
  const startDate = isJsonString(board.event?.startDate) ? board.event.startDate : null;
  const endDate = isJsonString(board.event?.endDate) ? board.event.endDate : null;
  const games = players.map((row) => ({
    gameId: row.gameId,
    league: "PGA",
    title: row.subject,
    homeTeam: null,
    awayTeam: null,
    gameDate: startDate,
    eventTime: row.eventTime,
    bookSpread: null,
    modelSpread: null,
    pointEdge: null,
    homeWinProb: row.modelProbability,
    modelVersion: row.modelVersion,
    freshnessStatus: row.marketStatus,
  }));
  const gaps = uniqueGaps([
    ...(board.gaps ?? []),
    games.length === 0 ? "Empty PGA tournament board." : null,
    "PGA rows are model win probabilities, not sportsbook prices.",
  ]);
  const modelVersion = isJsonString(board.predictionMeta?.model_version)
    ? board.predictionMeta.model_version
    : games[0]?.modelVersion ?? null;
  return {
    ...mcpMeta(
      board.generatedAt ?? null,
      board.dataSource === "supabase_predictions" ? "supabase" : board.dataSource === "static_json" ? "static_json" : "unavailable",
      gaps,
      modelVersion,
      "model-only",
    ),
    league,
    windowStart: startDate,
    windowEnd: endDate,
    games,
    event: { name: eventName, startDate, endDate },
  };
}

export async function handleGetGame(input: GetGameInput, deps: McpDeps = liveMcpDeps): Promise<GetGameResult> {
  const league = input.league;
  if (league !== "NBA" && league !== "NFL") {
    const gaps = [`get_game currently publishes line, win probability, and explanation features for NBA and NFL. ${league} is not a team-slate game lookup.`];
    return {
      ...mcpMeta(null, "unavailable", gaps, null, league === "MLB" ? "research" : "model-only"),
      game: null,
    };
  }

  const lookaheadDays = clampLookaheadDays(league);
  let gameId = input.id?.trim() ?? "";
  const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(gameId);
  const extraGaps: string[] = [];

  if (!uuid) {
    const feed = await teamSlate(league, lookaheadDays, deps);
    extraGaps.push(...feed.gaps);
    const matches = matchSlateGame(feed.games, input.id, input.team);
    if (matches.length > 1) extraGaps.push(`Multiple ${league} games matched ${input.id ?? input.team}; returning the first (${matches[0].awayTeam} @ ${matches[0].homeTeam}).`);
    gameId = matches[0]?.gameId ?? "";
  }

  if (!gameId) {
    const gaps = uniqueGaps([...extraGaps, `No ${league} game found for ${input.id ?? input.team ?? "the given lookup"}.`]);
    return { ...mcpMeta(null, "unavailable", gaps, PREFERRED_MODEL_VERSION[league], "research"), game: null };
  }

  const detail = await deps.getMobileGameDetail(league, gameId);
  if (!detail.data) {
    return {
      ...mcpMeta(detail.updatedAt, "unavailable", uniqueGaps([...extraGaps, ...detail.gaps]), PREFERRED_MODEL_VERSION[league], "research"),
      game: null,
    };
  }

  const game = gameDetailFromMobile(detail.data.game, detail.data.explanation);
  const gaps = uniqueGaps([
    ...extraGaps,
    ...detail.gaps,
    game.topFeatures.length === 0 ? "No explanation features published for this game." : null,
  ]);
  const status = publicationStatus({
    productionStatus: (await deps.getMobilePerformance()).data.records.find((row) => row.league === league)?.productionStatus,
  });
  return {
    ...mcpMeta(detail.updatedAt, "supabase", gaps, game.modelVersion, status),
    game,
  };
}

export async function handleGetResults(input: GetResultsInput, deps: McpDeps = liveMcpDeps): Promise<GetResultsResult> {
  const league = input.league.toUpperCase();
  const window = input.window;
  if (league === "PGA") {
    const result = await deps.getPgaResultRows();
    const rows = filterMcpResults(result.rows, window, "evaluated_at");
    const summaries = summarizePga(rows).map((row): McpResultsSummary => ({
      market: row.market,
      modelVersion: row.modelVersion,
      sample: row.sample,
      wins: row.wins,
      losses: row.losses,
      pushes: row.pushes,
      units: row.flatUnits ?? null,
      hitRate: row.hitRate,
      roi: row.roi,
    }));
    const asOf = latestTimestamp(rows.map((row) => row.evaluated_at));
    const gaps = uniqueGaps([
      ...result.gaps,
      rows.length === 0 ? `No graded PGA results for window ${window}.` : null,
      "PGA results are placement hits, not sportsbook units.",
    ]);
    return {
      ...mcpMeta(asOf, rows.length ? "supabase" : "unavailable", gaps, summaries[0]?.modelVersion ?? null, "model-only"),
      league,
      window,
      summaries,
      games: [],
    };
  }

  if (league === "MLB") {
    const [games, homeRuns] = await Promise.all([deps.getGameResultRows("MLB"), deps.getMlbHomeRunResultRows()]);
    const gameRows = filterMcpResults(games.rows, window, "game_date");
    const hrRows = filterMcpResults(homeRuns.rows, window, "game_date");
    const summaries = [
      ...summarizeGameResults(gameRows),
      ...summarizeMlbHr(hrRows),
    ].map((row): McpResultsSummary => ({
      market: row.market,
      modelVersion: row.modelVersion,
      sample: row.sample,
      wins: row.wins,
      losses: row.losses,
      pushes: row.pushes,
      units: row.flatUnits ?? null,
      hitRate: row.hitRate,
      roi: row.roi,
    }));
    const asOf = latestTimestamp([
      ...gameRows.map((row) => row.evaluated_at),
      ...hrRows.map((row) => row.evaluated_at),
    ]);
    const gaps = uniqueGaps([
      ...games.gaps,
      ...homeRuns.gaps,
      !gameRows.length && !hrRows.length ? `No graded MLB results for window ${window}.` : null,
    ]);
    return {
      ...mcpMeta(asOf, gameRows.length || hrRows.length ? "supabase" : "unavailable", gaps, summaries[0]?.modelVersion ?? null, "research"),
      league,
      window,
      summaries,
      games: gameRows.slice(0, MCP_MAX_RESULT_ROWS).map(resultGameFromRow),
    };
  }

  const result = await deps.getGameResultRows(league);
  const rows = filterMcpResults(result.rows, window, "game_date");
  const spreadUnits = rows.reduce((sum, game) => sum + (game.flat_ats_units ?? 0), 0);
  const summaries = summarizeGameResults(rows).map((row): McpResultsSummary => ({
    market: row.market,
    modelVersion: row.modelVersion,
    sample: row.sample,
    wins: row.wins,
    losses: row.losses,
    pushes: row.pushes,
    units: row.market === "spread" ? spreadUnits : null,
    hitRate: row.hitRate,
    roi: row.roi,
  }));
  const asOf = latestTimestamp(rows.map((row) => row.evaluated_at ?? row.game_date));
  const gaps = uniqueGaps([
    ...result.gaps,
    rows.length === 0 ? `No graded ${league} results for window ${window}. W-L is not invented.` : null,
    window === "last_sunday" ? `last_sunday is the most recent Sunday in America/Denver (${rows[0]?.game_date ?? "no games"}).` : null,
  ]);
  return {
    ...mcpMeta(asOf, rows.length ? "supabase" : "unavailable", gaps, summaries[0]?.modelVersion ?? null, publicationStatus({ productionStatus: league === "NBA" ? "candidate" : "blocked" })),
    league,
    window,
    summaries,
    games: rows.slice(0, MCP_MAX_RESULT_ROWS).map(resultGameFromRow),
  };
}

export async function handleGetModelPerformance(
  input: GetModelPerformanceInput = {},
  deps: McpDeps = liveMcpDeps,
): Promise<GetModelPerformanceResult> {
  const [history, bundle] = await Promise.all([deps.getMobilePerformance(), deps.getEvaluationsBundle()]);
  const league = input.league?.toUpperCase();
  const records = history.data.records
    .filter((row) => !league || row.league.toUpperCase() === league)
    .map((row) => ({
      league: row.league,
      modelVersion: row.modelVersion,
      season: row.season,
      market: row.market,
      sampleSize: row.sampleSize,
      roi: row.roi,
      units: row.units,
      bets: row.bets,
      wins: row.wins,
      losses: row.losses,
      productionStatus: row.productionStatus,
      gates: row.gates,
    }));
  const backtests = bundle.strategies
    .filter((row) => !league || row.league.toUpperCase() === league)
    .slice(0, 25)
    .map((row) => ({
      league: row.league,
      modelVersion: row.modelVersion,
      strategyId: row.strategyId,
      market: row.market,
      sampleSize: row.sampleSize,
      bets: row.bets,
      roi: row.roi,
    }));
  const gaps = uniqueGaps([
    ...history.gaps,
    ...bundle.gaps,
    records.length === 0 ? `No performance records for ${league ?? "any league"}.` : null,
    "ROI is often negative. These are research metrics, not a profitable betting system.",
  ]);
  const status = publicationStatus({ productionStatus: records[0]?.productionStatus ?? null });
  return {
    ...mcpMeta(history.updatedAt, history.source, gaps, records[0]?.modelVersion ?? null, status),
    records,
    backtests,
    note: "Sports Edge is a versioned prediction app. Negative or null ROI is expected on several surfaces; do not treat a number as a lock.",
  };
}

export async function handleGetPlayerMarkets(
  input: GetPlayerMarketsInput,
  deps: McpDeps = liveMcpDeps,
): Promise<GetPlayerMarketsResult> {
  if (input.league === "MLB") {
    const feed = await deps.getMlbHomeRunFeed();
    const rows = feed.predictions.slice(0, MCP_MAX_PLAYER_ROWS).map(publicPlayerMarket);
    const gaps = uniqueGaps([
      ...feed.gaps,
      "MLB home-run rows are model probabilities and derived fair odds. They are not sportsbook prices, and no book EV is calculated.",
      rows.length === 0 ? "No current MLB home-run probabilities." : null,
    ]);
    return {
      ...mcpMeta(feed.generatedAt, feed.dataSource === "supabase_predictions" ? "supabase" : feed.dataSource === "static_json" ? "static_json" : "unavailable", gaps, feed.modelVersion, "candidate"),
      league: "MLB",
      rows,
    };
  }

  const feed = await deps.getNflAnytimeTdFeed();
  const rows = feed.predictions.slice(0, MCP_MAX_PLAYER_ROWS).map(publicPlayerMarket);
  const gaps = uniqueGaps([
    ...feed.gaps,
    "NFL anytime-TD rows are model probabilities and derived fair odds. They are not sportsbook prices, and no book EV is calculated.",
    rows.length === 0 ? "No current NFL anytime-TD probabilities." : null,
  ]);
  return {
    ...mcpMeta(feed.generatedAt, rows.length ? "supabase" : "unavailable", gaps, rows[0]?.modelVersion ?? null, "model-only"),
    league: "NFL",
    rows,
  };
}

export {
  rankEdgeRows,
  matchSlateGame,
  publicPlayerMarket,
  topExplanationFeatures,
} from "@/lib/mcp/public-rows";
