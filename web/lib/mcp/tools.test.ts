import { describe, expect, it } from "vitest";

import type { GameExplanation } from "@/lib/data/explanations";
import type { MlbResearchBoardData } from "@/lib/data/mlb-research";
import type { NflAnytimeTdFeed } from "@/lib/data/nfl-anytime-td";
import type { MlbHomeRunFeed, PgaBoardData } from "@/lib/data/player-markets";
import type { GameResultRow, MlbHrResultRow, PgaResultRow, RawResults } from "@/lib/data/results";
import type { TeamSlateFeed, TeamSlateGame } from "@/lib/data/team-markets";
import type { Prediction } from "@/lib/data/types";
import type { MobileGameDetailData } from "@/lib/mobile/types";
import { publicationStatus } from "@/lib/mcp/envelope";
import {
  handleGetGame,
  handleGetPlayerMarkets,
  handleGetResults,
  handleGetStatus,
  handleListEdges,
  handleGetModelPerformance,
  type McpDeps,
} from "@/lib/mcp/handlers";
import { matchSlateGame, publicPlayerMarket, rankEdgeRows, topExplanationFeatures } from "@/lib/mcp/public-rows";
import type { McpEdgeRow } from "@/lib/mcp/types";
import { lastSundayDate } from "@/lib/mcp/windows";
import { takeRateLimitToken } from "@/lib/rate-limit";

const slateGame = (overrides: Partial<TeamSlateGame> = {}): TeamSlateGame => ({
  gameId: "82a44015-e8ac-4b16-be13-54c2c77bbe9a",
  league: "NBA",
  season: 2026,
  week: null,
  gameTimeUtc: "2026-11-28T00:00:00Z",
  gameDate: "2026-11-28",
  homeTeam: "HOU",
  awayTeam: "PHI",
  bookSpread: -12.5,
  modelSpread: -0.91,
  edgePts: 11.587644533814442,
  homeWinProb: 0.529,
  modelVersion: "v3",
  predictionTs: "2026-11-28T13:06:11.528Z",
  oddsTs: "2026-11-28T13:06:40.304Z",
  freshnessStatus: "fresh",
  injuryAdjusted: false,
  injuryDataMissing: false,
  ...overrides,
});

const emptyRaw = <T,>(): RawResults<T> => ({ rows: [], gaps: [] });

function deps(overrides: Partial<McpDeps> = {}): McpDeps {
  const nbaFeed: TeamSlateFeed = {
    league: "NBA",
    generatedAt: "2026-11-28T13:06:11.528Z",
    windowStart: "2026-11-28",
    windowEnd: "2026-11-29",
    games: [slateGame()],
    gaps: [],
  };
  const nflFeed: TeamSlateFeed = {
    league: "NFL",
    generatedAt: "2026-11-26T13:07:59.920Z",
    windowStart: "2026-11-24",
    windowEnd: "2026-11-30",
    games: [
      slateGame({
        gameId: "d0c6a8fa-f983-4680-a147-db09783de891",
        league: "NFL",
        homeTeam: "SF",
        awayTeam: "SEA",
        bookSpread: 1.5,
        edgePts: -6.46339745948277,
        modelVersion: "nfl-v2-live-20261020",
        predictionTs: "2026-11-26T13:07:59.920Z",
      }),
    ],
    gaps: [],
  };
  const mlbBoard: MlbResearchBoardData = {
    generatedAt: "2026-11-28T12:00:00Z",
    market: "moneyline",
    slateDate: "2026-11-28",
    predictions: [],
    gaps: ["No MLB moneyline research predictions available for 2026-11-28."],
    dataSource: "supabase",
  };
  const mlbHr: MlbHomeRunFeed = {
    generatedAt: null,
    modelVersion: "mlb-hr-v1-heuristic",
    productionStatus: "candidate",
    predictions: [],
    gaps: ["No current MLB home-run probabilities."],
    dataSource: "unavailable",
  };
  const pga: PgaBoardData = {
    generatedAt: "2026-11-20T00:00:00Z",
    dataSource: "static_json",
    gaps: [],
    event: { name: "Example Open", startDate: "2026-11-20", endDate: "2026-11-23" },
    predictions: [],
    normalizedMarkets: [],
  };
  const tdFeed: NflAnytimeTdFeed = {
    generatedAt: "2026-11-26T13:00:00Z",
    predictions: [
      {
        id: "nfl-td-1",
        sport: "NFL",
        league: "NFL",
        gameId: "game-td",
        eventTime: "2026-11-29T18:00:00Z",
        subject: "Example Runner TD",
        player: "Example Runner",
        homeTeam: "DEN",
        awayTeam: "KC",
        market: "anytime_td",
        book: "model",
        line: null,
        price: null,
        modelProbability: 0.32,
        impliedProbability: null,
        edge: null,
        ev: null,
        kelly: null,
        confidence: 0.7,
        modelVersion: "nfl-anytime-td-v1",
        marketStatus: "model_only",
        updatedAt: "2026-11-26T13:00:00Z",
      },
    ],
    gaps: ["NFL anytime-TD fair odds come from model probabilities."],
  };
  const performance = {
    data: {
      generatedAt: "2026-11-24T23:37:53.293Z",
      records: [
        {
          league: "NBA",
          modelVersion: "v3",
          season: "2025-26",
          market: "spread",
          sampleSize: 1175,
          roi: -0.075,
          units: null,
          bets: 64,
          wins: 31,
          losses: 33,
          pushes: 0,
          productionStatus: "candidate",
          gates: [{ id: "strategy", label: "Strategy ROI", status: "blocked", detail: "Best recorded ROI -7.5%." }],
        },
      ],
    },
    gaps: ["NBA: historical rows missing book_spread"],
    updatedAt: "2026-11-24T23:37:53.293Z",
    source: "static_json" as const,
  };
  const base: McpDeps = {
    getTeamSlateFeed: async (league) => (league === "NBA" ? nbaFeed : nflFeed),
    getTeamSlateGame: async () => slateGame(),
    getTeamMarketPredictions: async (league) => ({
      generatedAt: "2026-11-28T13:06:11.528Z",
      predictions: league === "NBA"
        ? [{
            id: "nba-ml-home",
            sport: "NBA",
            league: "NBA",
            gameId: slateGame().gameId,
            eventTime: slateGame().gameTimeUtc,
            subject: "HOU moneyline",
            homeTeam: "HOU",
            awayTeam: "PHI",
            market: "moneyline",
            book: "pinnacle",
            line: null,
            price: -180,
            modelProbability: 0.72,
            impliedProbability: 0.64,
            edge: 0.08,
            ev: 0.12,
            kelly: 0.03,
            confidence: 0.44,
            modelVersion: "v3",
            marketStatus: "research",
            updatedAt: "2026-11-28T13:06:11.528Z",
          } satisfies Prediction]
        : [],
      gaps: [],
    }),
    getMobileGameDetail: async () => ({
      data: {
        game: {
          id: "NBA-82a44015-e8ac-4b16-be13-54c2c77bbe9a",
          gameId: slateGame().gameId,
          league: "NBA",
          kind: "team_spread",
          title: "PHI @ HOU",
          subtitle: "2026-11-28",
          eventTime: slateGame().gameTimeUtc,
          homeTeam: "HOU",
          awayTeam: "PHI",
          subject: null,
          market: "spread",
          book: "market",
          line: -12.5,
          price: null,
          modelProbability: 0.529,
          impliedProbability: null,
          edge: 11.587644533814442,
          ev: null,
          confidence: 0.529,
          modelVersion: "v3",
          freshnessStatus: "fresh",
          predictionTs: slateGame().predictionTs,
          oddsTs: slateGame().oddsTs,
          injuryAdjusted: false,
          injuryDataMissing: false,
        },
        explanation: {
          gameId: slateGame().gameId,
          league: "NBA",
          modelVersion: "v3",
          predictionTs: slateGame().predictionTs ?? "2026-11-28T13:06:11.528Z",
          topFeatures: [
            { feature: "net_rating_diff", value: 8.4, impact: 4.2 },
            { feature: "rest_days", value: 1, impact: 1.1 },
          ],
          injuryAdjusted: false,
          homeInjuryDelta: null,
          awayInjuryDelta: null,
          baseVsAdjusted: null,
        },
      } satisfies MobileGameDetailData,
      gaps: [],
      updatedAt: slateGame().predictionTs,
      source: "supabase" as const,
    }),
    getMobilePerformance: async () => performance,
    getMlbResearchBoard: async () => mlbBoard,
    getMlbHomeRunFeed: async () => mlbHr,
    getPgaBoardData: async () => pga,
    getNflAnytimeTdFeed: async () => tdFeed,
    getGameResultRows: async () => emptyRaw<GameResultRow>(),
    getMlbHomeRunResultRows: async () => emptyRaw<MlbHrResultRow>(),
    getPgaResultRows: async () => emptyRaw<PgaResultRow>(),
    getEvaluationsBundle: async () => ({
      generatedAt: "2026-11-28T00:00:00Z",
      evaluations: [],
      strategies: [],
      registry: [],
      gaps: [],
    }),
  };
  return { ...base, ...overrides };
}

describe("MCP ranking and sanitizing", () => {
  it("ranks by absolute point edge and keeps the requested sort", () => {
    const rows: McpEdgeRow[] = [
      { id: "a", league: "NBA", title: "BKN @ CHA", homeTeam: "CHA", awayTeam: "BKN", market: "spread", line: -5.5, modelProbability: 0.42, pointEdge: 8.8, ev: null, modelVersion: "v3", status: "candidate" },
      { id: "b", league: "NBA", title: "PHI @ HOU", homeTeam: "HOU", awayTeam: "PHI", market: "spread", line: -12.5, modelProbability: 0.53, pointEdge: 11.587644533814442, ev: null, modelVersion: "v3", status: "candidate" },
      { id: "c", league: "NFL", title: "priced", homeTeam: "KC", awayTeam: "DEN", market: "moneyline", line: null, modelProbability: 0.6, pointEdge: null, ev: 0.2, modelVersion: "v1", status: "research" },
    ];
    const ranked = rankEdgeRows(rows, "point_edge", 8);
    expect(ranked[0]?.title).toBe("PHI @ HOU");
    expect(ranked.map((row) => row.id)).toEqual(["b", "a"]);
  });

  it("EV sort drops rows without a sportsbook EV", () => {
    const rows: McpEdgeRow[] = [
      { id: "model-only", league: "NFL", title: "TD", homeTeam: null, awayTeam: null, market: "anytime_td", line: null, modelProbability: 0.3, pointEdge: null, ev: null, modelVersion: "td", status: "model-only" },
      { id: "priced", league: "NBA", title: "HOU ML", homeTeam: "HOU", awayTeam: "PHI", market: "moneyline", line: null, modelProbability: 0.7, pointEdge: null, ev: 0.12, modelVersion: "v3", status: "research" },
    ];
    expect(rankEdgeRows(rows, "ev", 8).map((row) => row.id)).toEqual(["priced"]);
  });

  it("strips book EV and raw features from player markets", () => {
    const row = publicPlayerMarket({
      id: "hr-1",
      sport: "MLB",
      league: "MLB",
      gameId: "g",
      eventTime: null,
      subject: "Player HR",
      player: "Player",
      market: "home_run",
      book: "model",
      line: 0.5,
      price: 400,
      modelProbability: 0.2,
      impliedProbability: 0.15,
      edge: 0.05,
      ev: 0.4,
      kelly: 0.02,
      confidence: 0.5,
      modelVersion: "mlb-hr-v1",
      marketStatus: "model_only",
      updatedAt: "2026-11-28T12:00:00Z",
    });
    expect(row.modelProbability).toBe(0.2);
    expect(row.fairOdds).toMatch(/^\+/);
    expect(row).not.toHaveProperty("ev");
    expect(row).not.toHaveProperty("edge");
    expect(row).not.toHaveProperty("topFeatures");
  });

  it("keeps only the top explanation features", () => {
    const explanation: GameExplanation = {
      gameId: "g",
      league: "NBA",
      modelVersion: "v3",
      predictionTs: "2026-11-28T13:06:11Z",
      topFeatures: [
        { feature: "noise", value: 1, impact: 0.1 },
        { feature: "net_rating_diff", value: 8.4, impact: 4.2 },
      ],
      injuryAdjusted: false,
      homeInjuryDelta: null,
      awayInjuryDelta: null,
      baseVsAdjusted: { raw: true },
    };
    expect(topExplanationFeatures(explanation)[0]).toMatchObject({ feature: "net_rating_diff", impact: 4.2 });
  });
});

describe("MCP tool handlers", () => {
  it("list_edges names point_edge sort and returns the fixture's biggest NBA edge", async () => {
    const result = await handleListEdges({ sort: "point_edge", limit: 8 }, deps());
    expect(result.sort).toBe("point_edge");
    expect(result.rows[0]?.title).toBe("PHI @ HOU");
    expect(result.rows[0]?.pointEdge).toBeCloseTo(11.587644533814442);
    expect(result.as_of).toBeTruthy();
    expect(result.model_version).toContain("v3");
    expect(result.freshness).toBeDefined();
    expect(result.status).toBeTruthy();
    expect(result.gaps).toBeDefined();
  });

  it("list_edges EV sort reports ev and excludes unpriced rows", async () => {
    const result = await handleListEdges({ league: "NBA", sort: "ev", limit: 8 }, deps());
    expect(result.sort).toBe("ev");
    expect(result.rows[0]?.ev).toBe(0.12);
    expect(result.rows[0]?.title).toContain("HOU");
  });

  it("get_game by team cites explanation features", async () => {
    const result = await handleGetGame({ league: "NBA", team: "HOU" }, deps());
    expect(result.game?.title).toBe("PHI @ HOU");
    expect(result.game?.topFeatures.some((feature) => feature.feature === "net_rating_diff")).toBe(true);
    expect(result.game).not.toHaveProperty("baseVsAdjusted");
  });

  it("get_results last_sunday uses graded rows and does not invent W-L", async () => {
    const sunday = lastSundayDate(new Date("2026-11-28T18:00:00Z"));
    expect(sunday).toBe("2026-11-22");
    const rows: GameResultRow[] = [
      {
        league: "NFL",
        season: 2026,
        week: 11,
        game_date: "2026-11-22",
        home_team: "KC",
        away_team: "DEN",
        home_score: 27,
        away_score: 20,
        book_spread: -7,
        my_spread: -4,
        my_home_win_prob: 0.61,
        model_version: "nfl-v2-live-20261020",
        spread_result: "loss",
        winner_result: "win",
        flat_ats_units: -1,
        evaluated_at: "2026-11-23T06:00:00Z",
      },
      {
        league: "NFL",
        season: 2026,
        week: 11,
        game_date: "2026-11-22",
        home_team: "BUF",
        away_team: "HOU",
        home_score: 24,
        away_score: 17,
        book_spread: -3,
        my_spread: -5,
        my_home_win_prob: 0.58,
        model_version: "nfl-v2-live-20261020",
        spread_result: "win",
        winner_result: "win",
        flat_ats_units: 0.91,
        evaluated_at: "2026-11-23T06:00:00Z",
      },
      {
        league: "NFL",
        season: 2026,
        week: 10,
        game_date: "2026-11-15",
        home_team: "DAL",
        away_team: "NYG",
        home_score: 30,
        away_score: 10,
        book_spread: -7,
        my_spread: -3,
        my_home_win_prob: 0.55,
        model_version: "nfl-v2-live-20261020",
        spread_result: "win",
        winner_result: "win",
        flat_ats_units: 0.91,
        evaluated_at: "2026-11-16T06:00:00Z",
      },
    ];
    const result = await handleGetResults(
      { league: "NFL", window: "last_sunday" },
      deps({
        getGameResultRows: async () => ({ rows, gaps: [] }),
      }),
    );
    const spread = result.summaries.find((row) => row.market === "spread");
    expect(result.window).toBe("last_sunday");
    expect(result.games).toHaveLength(2);
    expect(spread).toMatchObject({ wins: 1, losses: 1, sample: 2 });
    expect(result.games.every((game) => game.gameDate === "2026-11-22")).toBe(true);
    expect(spread?.wins).not.toBe(16);
  });

  it("get_status discloses a stale league", async () => {
    const stale = deps({
      getTeamSlateFeed: async (league) => {
        if (league === "NFL") {
          return {
            league: "NFL",
            generatedAt: "2026-11-20T13:00:00Z",
            windowStart: "2026-11-24",
            windowEnd: "2026-11-30",
            games: [
              slateGame({
                league: "NFL",
                homeTeam: "SF",
                awayTeam: "SEA",
                predictionTs: "2026-11-01T13:00:00Z",
                modelVersion: "nfl-v2-live-20261020",
                freshnessStatus: "stale",
              }),
            ],
            gaps: ["NFL predictions are older than the current cycle."],
          };
        }
        return {
          league: "NBA",
          generatedAt: "2026-11-28T13:06:11.528Z",
          windowStart: "2026-11-28",
          windowEnd: "2026-11-29",
          games: [slateGame()],
          gaps: [],
        };
      },
    });
    const result = await handleGetStatus(stale);
    const nfl = result.leagues.find((row) => row.league === "NFL");
    expect(nfl?.freshness.status).toBe("stale");
    expect(result.gaps.join(" ")).toMatch(/stale|older|missing/i);
  });

  it("player markets stay model-only without fake book EV", async () => {
    const result = await handleGetPlayerMarkets({ league: "NFL" }, deps());
    expect(result.status).toBe("model-only");
    expect(result.rows[0]?.modelProbability).toBe(0.32);
    expect(result.gaps.join(" ")).toMatch(/not sportsbook/i);
    expect(JSON.stringify(result)).not.toMatch(/"ev":/);
  });

  it("model performance reports negative ROI honestly", async () => {
    const result = await handleGetModelPerformance({ league: "NBA" }, deps());
    expect(result.records[0]?.roi).toBeLessThan(0);
    expect(result.note).toMatch(/do not treat a number as a lock/i);
  });

  it("matchSlateGame finds a team abbreviation", () => {
    expect(matchSlateGame([slateGame()], undefined, "hou")[0]?.homeTeam).toBe("HOU");
  });
});

describe("rate limiter", () => {
  it("allows traffic under the cap and then returns 429 semantics", () => {
    const store = new Map();
    const now = 1_000_000;
    for (let i = 0; i < 60; i += 1) {
      expect(takeRateLimitToken(store, "ip", now, 60, 60_000).allowed).toBe(true);
    }
    const blocked = takeRateLimitToken(store, "ip", now, 60, 60_000);
    expect(blocked.allowed).toBe(false);
    expect(blocked.retryAfterSeconds).toBeGreaterThan(0);
  });
});

describe("publication status", () => {
  it("maps model-only, candidate, and blocked honestly", () => {
    expect(publicationStatus({ marketStatus: "model_only" })).toBe("model-only");
    expect(publicationStatus({ productionStatus: "candidate" })).toBe("candidate");
    expect(publicationStatus({ productionStatus: "blocked" })).toBe("research");
    expect(publicationStatus({ productionStatus: "approved" })).toBe("production");
  });
});
