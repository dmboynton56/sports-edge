import { NextResponse } from "next/server";

import { envelope, getMobileGameDetail } from "@/lib/mobile/normalize";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ league: string; id: string }> },
) {
  const values = await params;
  const league = values.league.toUpperCase();
  if (league !== "NBA" && league !== "NFL") {
    return NextResponse.json({ error: "Game detail is currently available for NBA and NFL." }, { status: 400 });
  }
  const result = await getMobileGameDetail(league, values.id);
  const status = result.data ? 200 : 404;
  return NextResponse.json(envelope(result.data, result.gaps, result.updatedAt, result.source), { status });
}
