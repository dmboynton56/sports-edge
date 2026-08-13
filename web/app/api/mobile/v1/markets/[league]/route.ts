import { NextResponse } from "next/server";

import { envelope, getMobileMarkets } from "@/lib/mobile/normalize";
import type { MobileLeague } from "@/lib/mobile/types";

export const dynamic = "force-dynamic";

const LEAGUES = new Set<MobileLeague>(["NBA", "NFL", "MLB", "PGA"]);

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ league: string }> },
) {
  const league = (await params).league.toUpperCase() as MobileLeague;
  if (!LEAGUES.has(league)) {
    return NextResponse.json({ error: "Unsupported league", supported: [...LEAGUES] }, { status: 400 });
  }
  const result = await getMobileMarkets(league);
  return NextResponse.json(envelope(result.data, result.gaps, result.updatedAt, result.source));
}
