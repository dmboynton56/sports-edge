import { freshnessFor } from "@/lib/mobile/normalize";
import type { MobileSource } from "@/lib/mobile/types";
import type { McpMeta, McpPublicationStatus } from "@/lib/mcp/types";

export function uniqueGaps(gaps: (string | null | undefined)[]): string[] {
  return Array.from(new Set(gaps.filter((gap): gap is string => Boolean(gap))));
}

export function publicationStatus(args: {
  marketStatus?: string | null;
  productionStatus?: string | null;
  registryStatus?: string | null;
}): McpPublicationStatus {
  const market = (args.marketStatus ?? "").toLowerCase();
  if (market === "model_only" || market === "model-only") return "model-only";
  if (market === "research") return "research";

  const production = (args.productionStatus ?? "").toLowerCase();
  if (production === "blocked") return "research";
  if (production === "candidate") return "candidate";
  if (production === "approved") return "production";

  const registry = (args.registryStatus ?? "").toLowerCase();
  if (registry === "production") return "production";
  if (registry === "candidate") return "candidate";
  if (registry === "research") return "research";
  if (registry === "model-only" || registry === "model_only") return "model-only";
  return "research";
}

export function mcpMeta(
  updatedAt: string | null,
  source: MobileSource,
  gaps: string[],
  modelVersion: string | null,
  status: McpPublicationStatus,
): McpMeta {
  return {
    as_of: updatedAt,
    model_version: modelVersion,
    freshness: freshnessFor(updatedAt, source, gaps),
    status,
    gaps: uniqueGaps(gaps),
  };
}

export function latestTimestamp(values: (string | null | undefined)[]): string | null {
  const stamps = values.filter((value): value is string => Boolean(value)).sort();
  return stamps.at(-1) ?? null;
}

export function toolJsonResult(payload: unknown) {
  return {
    content: [
      {
        type: "text" as const,
        text: JSON.stringify(payload, null, 2),
      },
    ],
  };
}

export function toolErrorResult(message: string) {
  return {
    isError: true,
    content: [{ type: "text" as const, text: message }],
  };
}
