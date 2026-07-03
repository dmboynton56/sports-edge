import { NextResponse } from "next/server";

import { getPgaBoardData } from "@/lib/data/player-markets";

export const dynamic = "force-dynamic";

export async function GET() {
  const board = await getPgaBoardData();
  return NextResponse.json(board);
}
