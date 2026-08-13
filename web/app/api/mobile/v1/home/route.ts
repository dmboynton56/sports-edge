import { NextResponse } from "next/server";

import { envelope, getMobileHome } from "@/lib/mobile/normalize";

export const dynamic = "force-dynamic";

export async function GET() {
  const result = await getMobileHome();
  return NextResponse.json(envelope(result.data, result.gaps, result.updatedAt, result.source));
}
