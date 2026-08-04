import { NextResponse } from "next/server";

import { getFantasyFeed } from "@/lib/data/fantasy-server";

export async function GET() {
  const feed = await getFantasyFeed("preseason");
  return NextResponse.json({
    generatedAt: feed.generatedAt,
    season: feed.season,
    source: "FantasyPros ADP market signal",
    adp: feed.adp,
    gaps: feed.gaps,
  });
}
