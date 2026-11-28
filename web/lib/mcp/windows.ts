import { denverDate } from "@/lib/nfl-week";
import { filterByWindow, type ResultsWindow } from "@/lib/data/results";
import { isJsonString } from "@/lib/data/json";
import type { McpLeague, McpResultsWindow } from "@/lib/mcp/types";
import { MCP_MAX_EDGE_LIMIT, MCP_MAX_LOOKAHEAD_DAYS } from "@/lib/mcp/types";

export function clampEdgeLimit(limit: number): number {
  if (!Number.isFinite(limit) || limit <= 0) return 8;
  return Math.min(MCP_MAX_EDGE_LIMIT, Math.trunc(limit));
}

export function clampLookaheadDays(league: McpLeague, requested?: number): number {
  const max = MCP_MAX_LOOKAHEAD_DAYS[league];
  if (requested == null || !Number.isFinite(requested) || requested <= 0) return max;
  return Math.min(max, Math.trunc(requested));
}

export function lastSundayDate(now = new Date()): string {
  const today = denverDate(now);
  const date = new Date(`${today}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() - date.getUTCDay());
  return date.toISOString().slice(0, 10);
}

export function filterMcpResults<T>(
  rows: T[],
  window: McpResultsWindow,
  dateField: keyof T,
  now = new Date(),
): T[] {
  if (window === "last_sunday") {
    const sunday = lastSundayDate(now);
    return rows.filter((row) => {
      const value = row[dateField];
      if (!isJsonString(value)) return false;
      return value.slice(0, 10) === sunday;
    });
  }
  const mapped: ResultsWindow = window;
  return filterByWindow(rows, mapped, dateField, now);
}
