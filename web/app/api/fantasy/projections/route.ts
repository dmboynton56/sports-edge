import { NextResponse } from "next/server";

import { getFantasyFeed } from "@/lib/data/fantasy";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const scope = url.searchParams.get("scope") === "week" ? "week" : "preseason";
  const weekValue = Number(url.searchParams.get("week") ?? "1");
  const week = Number.isFinite(weekValue) ? Math.min(18, Math.max(1, weekValue)) : 1;
  const feed = await getFantasyFeed(scope, week);
  return NextResponse.json({
    generatedAt: feed.generatedAt,
    season: feed.season,
    modelVersion: feed.modelVersion,
    productionStatus: feed.productionStatus,
    defaultScoring: feed.defaultScoring,
    projections: feed.projections,
    gaps: feed.gaps,
    sources: feed.sources,
  });
}
