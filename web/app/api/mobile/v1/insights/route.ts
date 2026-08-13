import { NextResponse } from "next/server";

import { envelope, getMobileInsights } from "@/lib/mobile/normalize";

export const dynamic = "force-dynamic";

export async function GET() {
  const result = await getMobileInsights();
  return NextResponse.json(envelope(result.data, result.gaps, result.updatedAt, result.source));
}
