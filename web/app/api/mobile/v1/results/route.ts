import { NextResponse } from "next/server";

import { handleGetResults } from "@/lib/mcp/handlers";
import { envelope } from "@/lib/mobile/normalize";

export const dynamic = "force-dynamic";

const WINDOWS = { last_sunday: true, "7d": true, season: true } as const;

function isWindow(value: string): value is keyof typeof WINDOWS {
  return Object.keys(WINDOWS).includes(value);
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const league = url.searchParams.get("league")?.trim().toUpperCase();
  const window = url.searchParams.get("window")?.trim() ?? "7d";
  if (!league) {
    return NextResponse.json({ error: "league is required" }, { status: 400 });
  }
  if (!isWindow(window)) {
    return NextResponse.json({ error: "Unsupported window", supported: Object.keys(WINDOWS) }, { status: 400 });
  }
  const result = await handleGetResults({ league, window });
  return NextResponse.json(envelope(result, result.gaps, result.as_of, result.freshness.source));
}
