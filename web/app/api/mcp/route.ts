import { createMcpHandler } from "mcp-handler";
import { z } from "zod";

import {
  handleGetGame,
  handleGetModelPerformance,
  handleGetPlayerMarkets,
  handleGetResults,
  handleGetSlate,
  handleGetStatus,
  handleListEdges,
} from "@/lib/mcp/handlers";
import { toolErrorResult, toolJsonResult } from "@/lib/mcp/envelope";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const leagueSchema = z.enum(["NBA", "NFL", "MLB", "PGA"]);
const playerLeagueSchema = z.enum(["MLB", "NFL"]);
const sortSchema = z.enum(["point_edge", "ev"]);
const windowSchema = z.enum(["last_sunday", "7d", "season"]);

const handler = createMcpHandler(
  (server) => {
    server.registerTool(
      "get_status",
      {
        title: "Sports Edge status",
        description:
          "Freshness, gaps, and last publish per league. Read-only model output. Cite as_of, model_version, and research vs production. Never recommend a bet.",
        inputSchema: z.object({}),
      },
      async () => toolJsonResult(await handleGetStatus()),
    );

    server.registerTool(
      "list_edges",
      {
        title: "Ranked edges",
        description:
          "Ranked board rows. sort=point_edge (default) uses spread point differentials for NBA/NFL. sort=ev uses sportsbook-priced EV only and never invents EV from model fair odds. The response states which sort was used. Cap 25.",
        inputSchema: z.object({
          league: leagueSchema.optional(),
          sort: sortSchema.default("point_edge"),
          limit: z.number().int().min(1).max(25).default(8),
        }),
      },
      async ({ league, sort, limit }) => toolJsonResult(await handleListEdges({ league, sort, limit })),
    );

    server.registerTool(
      "get_slate",
      {
        title: "League slate",
        description:
          "Current slate for NBA, NFL, MLB, or PGA. Date windows are capped (NBA 2d, NFL 7d, MLB 1d, PGA current tournament). Empty slates are disclosed.",
        inputSchema: z.object({
          league: leagueSchema,
          lookahead_days: z.number().int().min(1).max(7).optional(),
        }),
      },
      async ({ league, lookahead_days: lookaheadDays }) =>
        toolJsonResult(await handleGetSlate({ league, lookaheadDays })),
    );

    server.registerTool(
      "get_game",
      {
        title: "Game detail",
        description:
          "NBA/NFL game line, win probability, top explanation features, and injury flags. Look up by game id or team abbreviation. Does not expose raw feature vectors or model_runs.",
        inputSchema: z.object({
          league: leagueSchema,
          id: z.string().min(1).optional(),
          team: z.string().min(1).optional(),
        }),
      },
      async ({ league, id, team }) => {
        if (!id && !team) return toolErrorResult("Provide id or team.");
        return toolJsonResult(await handleGetGame({ league, id, team }));
      },
    );

    server.registerTool(
      "get_results",
      {
        title: "Graded results",
        description:
          "Graded W-L, units, and game rows from published results. window is last_sunday, 7d, or season. Never invent W-L. last_sunday is the most recent Sunday in America/Denver.",
        inputSchema: z.object({
          league: z.string().min(2).max(8),
          window: windowSchema,
        }),
      },
      async ({ league, window }) => toolJsonResult(await handleGetResults({ league, window })),
    );

    server.registerTool(
      "get_model_performance",
      {
        title: "Model performance",
        description:
          "Metrics, production gates, backtests, and ROI. ROI is often negative. This is research telemetry, not a betting system.",
        inputSchema: z.object({
          league: z.string().min(2).max(8).optional(),
        }),
      },
      async ({ league }) => toolJsonResult(await handleGetModelPerformance({ league })),
    );

    server.registerTool(
      "get_player_markets",
      {
        title: "Player model probabilities",
        description:
          "MLB home-run or NFL anytime-TD model probabilities and derived fair odds. Not sportsbook prices. No book EV or Kelly.",
        inputSchema: z.object({
          league: playerLeagueSchema,
        }),
      },
      async ({ league }) => toolJsonResult(await handleGetPlayerMarkets({ league })),
    );
  },
  {
    serverInfo: {
      name: "sports-edge",
      version: "1.0.0",
    },
    instructions:
      "Sports Edge is a read-only, versioned sports prediction app. It is not a betting system. Always cite as_of, model_version, freshness, and research vs production. Never recommend a bet or imply a lock. Fetch live; never quote numbers from memory.",
  },
);

export { handler as GET, handler as POST, handler as DELETE };
