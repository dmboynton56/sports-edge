export {
  handleGetGame,
  handleGetModelPerformance,
  handleGetPlayerMarkets,
  handleGetResults,
  handleGetSlate,
  handleGetStatus,
  handleListEdges,
  liveMcpDeps,
  rankEdgeRows,
  matchSlateGame,
  publicPlayerMarket,
} from "@/lib/mcp/handlers";
export { mcpMeta, publicationStatus, toolJsonResult, toolErrorResult } from "@/lib/mcp/envelope";
export { clampEdgeLimit, lastSundayDate, filterMcpResults } from "@/lib/mcp/windows";
export type { McpDeps } from "@/lib/mcp/handlers";
