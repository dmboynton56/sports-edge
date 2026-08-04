import { promises as fs } from "fs";
import path from "path";

import { getSupabaseMissingEnv, supabaseRest } from "@/lib/data/supabase";
import {
  DEFAULT_FANTASY_SCORING,
  fromStatic,
  normalizeProjection,
  type FantasyFeed,
  type SupabaseFantasyRow,
} from "@/lib/data/fantasy";

const FANTASY_PATH = path.join(process.cwd(), "public", "data", "fantasy_projections.json");

export async function getFantasyFeed(scope: "preseason" | "week" = "preseason", week = 1): Promise<FantasyFeed> {
  const season = new Date().getUTCFullYear();
  const missing = getSupabaseMissingEnv();
  const query = `?select=*&season=eq.${season}&scope=eq.${scope}&week=eq.${scope === "week" ? week : 0}&order=points.desc&limit=5000`;
  const rows = await supabaseRest<SupabaseFantasyRow>(`fantasy_player_projections_latest${query}`);
  if (rows?.length) {
    return {
      generatedAt: rows[0]?.updated_at ?? null,
      season,
      modelVersion: rows[0]?.model_version ?? "fantasy-v1",
      productionStatus: "candidate",
      defaultScoring: DEFAULT_FANTASY_SCORING,
      projections: scope === "preseason" ? rows.map(normalizeProjection) : [],
      weekly: scope === "week" ? { [String(week)]: rows.map(normalizeProjection) } : {},
      adp: [],
      metrics: {},
      gaps: [],
      sources: ["Supabase fantasy projections"],
      dataSource: "supabase",
    };
  }

  try {
    const payload = JSON.parse(await fs.readFile(FANTASY_PATH, "utf8")) as Partial<FantasyFeed>;
    const feed = fromStatic(payload, missing.length ? [`Supabase live feed unavailable: ${missing.join(", ")}.`] : []);
    if (scope === "week") {
      feed.projections = feed.weekly[String(week)] ?? [];
    }
    return feed;
  } catch {
    return {
      generatedAt: null,
      season,
      modelVersion: "fantasy-unavailable",
      productionStatus: "blocked",
      defaultScoring: DEFAULT_FANTASY_SCORING,
      projections: [],
      weekly: {},
      adp: [],
      metrics: {},
      gaps: [...missing.map((item) => `Missing ${item}.`), `No fantasy artifact found at ${FANTASY_PATH}.`],
      sources: [],
      dataSource: "unavailable",
    };
  }
}
