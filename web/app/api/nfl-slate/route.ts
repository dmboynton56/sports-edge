import { NextResponse } from "next/server";

import { getTeamSlateFeed } from "@/lib/data/team-markets";

export const dynamic = "force-dynamic";

export async function GET() {
  const feed = await getTeamSlateFeed("NFL", { lookaheadDays: 7 });
  return NextResponse.json(feed);
}
