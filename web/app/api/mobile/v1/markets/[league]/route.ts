import { NextResponse } from "next/server";

import { envelope, getMobileMarkets } from "@/lib/mobile/normalize";
import type { MobileLeague } from "@/lib/mobile/types";

export const dynamic = "force-dynamic";

const LEAGUES = { NBA: true, NFL: true, MLB: true, PGA: true } as const;

function isMobileLeague(value: string): value is MobileLeague {
  return Object.keys(LEAGUES).includes(value);
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ league: string }> },
) {
  const value = (await params).league.toUpperCase();
  if (!isMobileLeague(value)) {
    return NextResponse.json({ error: "Unsupported league", supported: Object.keys(LEAGUES) }, { status: 400 });
  }
  const league = value;
  const result = await getMobileMarkets(league);
  return NextResponse.json(envelope(result.data, result.gaps, result.updatedAt, result.source));
}
