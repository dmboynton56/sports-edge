import { NextResponse } from "next/server";

import { getTeamSlateFeed } from "@/lib/data/team-markets";

export const dynamic = "force-dynamic";

export async function GET() {
  const feed = await getTeamSlateFeed("NBA", { lookaheadDays: 1 });
  return NextResponse.json(feed);
}
