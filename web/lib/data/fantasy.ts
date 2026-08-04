import { promises as fs } from "fs";
import path from "path";

import { getSupabaseMissingEnv, supabaseRest } from "@/lib/data/supabase";

export type FantasyPosition = "QB" | "RB" | "WR" | "TE" | "K" | "DST";

export type FantasyScoring = {
  name: string;
  passing_yards: number;
  passing_td: number;
  interception: number;
  rushing_yards: number;
  rushing_td: number;
  reception: number;
  receiving_yards: number;
  receiving_td: number;
  fumble_lost: number;
  two_point_conversion: number;
  kick_extra_point: number;
  field_goal_0_39: number;
  field_goal_40_49: number;
  field_goal_50_plus: number;
  missed_field_goal: number;
  dst_sack: number;
  dst_interception: number;
  dst_fumble_recovery: number;
  dst_td: number;
  dst_safety: number;
  dst_blocked_kick: number;
  dst_points_0: number;
  dst_points_1_6: number;
  dst_points_7_13: number;
  dst_points_14_20: number;
  dst_points_21_27: number;
  dst_points_28_34: number;
  dst_points_35_plus: number;
};

export type FantasyRoster = {
  teams: number;
  quarterback: number;
  running_back: number;
  wide_receiver: number;
  tight_end: number;
  flex: number;
  kicker: number;
  defense: number;
  bench: number;
};

export type FantasyProjection = {
  player_id: string;
  player_name: string;
  position: FantasyPosition;
  team: string | null;
  season: number;
  scope: "preseason" | "week";
  week: number;
  projected_games: number;
  statline: Record<string, number>;
  statline_low?: Record<string, number>;
  statline_high?: Record<string, number>;
  points: number;
  floor_points: number;
  ceiling_points: number;
  points_per_game: number;
  overall_rank?: number;
  position_rank?: number;
  tier?: number;
  adp?: number | null;
  adp_rank?: number | null;
  adp_tier?: number | null;
  adp_source?: string | null;
  confidence: "high" | "medium" | "low";
  availability: string;
  explanation: string[];
  model_version: string;
  updated_at: string;
};

export type FantasyFeed = {
  generatedAt: string | null;
  season: number;
  modelVersion: string;
  productionStatus: "candidate" | "approved" | "blocked";
  defaultScoring: FantasyScoring;
  projections: FantasyProjection[];
  weekly: Record<string, FantasyProjection[]>;
  adp: Record<string, unknown>[];
  metrics: Record<string, unknown>;
  gaps: string[];
  sources: string[];
  dataSource: "supabase" | "static_json" | "unavailable";
};

type SupabaseFantasyRow = Partial<FantasyProjection> & {
  player_id: string;
  player_name: string;
  position: FantasyPosition;
  season: number;
  scope: "preseason" | "week";
  week: number;
  statline: Record<string, number> | null;
  statline_low: Record<string, number> | null;
  statline_high: Record<string, number> | null;
};

const FANTASY_PATH = path.join(process.cwd(), "public", "data", "fantasy_projections.json");

export const DEFAULT_FANTASY_SCORING: FantasyScoring = {
  name: "Full PPR",
  passing_yards: 0.04,
  passing_td: 4,
  interception: -2,
  rushing_yards: 0.1,
  rushing_td: 6,
  reception: 1,
  receiving_yards: 0.1,
  receiving_td: 6,
  fumble_lost: -2,
  two_point_conversion: 2,
  kick_extra_point: 1,
  field_goal_0_39: 3,
  field_goal_40_49: 4,
  field_goal_50_plus: 5,
  missed_field_goal: 0,
  dst_sack: 1,
  dst_interception: 2,
  dst_fumble_recovery: 2,
  dst_td: 6,
  dst_safety: 2,
  dst_blocked_kick: 2,
  dst_points_0: 10,
  dst_points_1_6: 7,
  dst_points_7_13: 4,
  dst_points_14_20: 1,
  dst_points_21_27: 0,
  dst_points_28_34: -1,
  dst_points_35_plus: -4,
};

export const HALF_PPR_SCORING: FantasyScoring = { ...DEFAULT_FANTASY_SCORING, name: "Half PPR", reception: 0.5 };
export const STANDARD_SCORING: FantasyScoring = { ...DEFAULT_FANTASY_SCORING, name: "Standard", reception: 0 };

export const DEFAULT_FANTASY_ROSTER: FantasyRoster = {
  teams: 12,
  quarterback: 1,
  running_back: 2,
  wide_receiver: 2,
  tight_end: 1,
  flex: 1,
  kicker: 1,
  defense: 1,
  bench: 6,
};

function unique(values: string[]) {
  return [...new Set(values.filter(Boolean))];
}

function normalizeProjection(row: SupabaseFantasyRow): FantasyProjection {
  return {
    player_id: String(row.player_id),
    player_name: String(row.player_name),
    position: row.position,
    team: row.team ?? null,
    season: Number(row.season),
    scope: row.scope,
    week: Number(row.week ?? 0),
    projected_games: Number(row.projected_games ?? 0),
    statline: row.statline ?? {},
    statline_low: row.statline_low ?? {},
    statline_high: row.statline_high ?? {},
    points: Number(row.points ?? 0),
    floor_points: Number(row.floor_points ?? 0),
    ceiling_points: Number(row.ceiling_points ?? 0),
    points_per_game: Number(row.points_per_game ?? 0),
    overall_rank: row.overall_rank == null ? undefined : Number(row.overall_rank),
    position_rank: row.position_rank == null ? undefined : Number(row.position_rank),
    tier: row.tier == null ? undefined : Number(row.tier),
    adp: row.adp == null ? null : Number(row.adp),
    adp_rank: row.adp_rank == null ? null : Number(row.adp_rank),
    adp_tier: row.adp_tier == null ? null : Number(row.adp_tier),
    adp_source: row.adp_source ?? null,
    confidence: row.confidence ?? "low",
    availability: row.availability ?? "expected",
    explanation: row.explanation ?? [],
    model_version: row.model_version ?? "unknown",
    updated_at: row.updated_at ?? new Date().toISOString(),
  };
}

function fromStatic(payload: Partial<FantasyFeed>, fallbackGaps: string[]): FantasyFeed {
  const preseason = (payload.projections ?? []).map((row) => normalizeProjection(row as SupabaseFantasyRow));
  const preseasonById = new Map(preseason.map((row) => [row.player_id, row]));
  return {
    generatedAt: payload.generatedAt ?? null,
    season: Number(payload.season ?? new Date().getUTCFullYear()),
    modelVersion: payload.modelVersion ?? "fantasy-unavailable",
    productionStatus: payload.productionStatus ?? "candidate",
    defaultScoring: payload.defaultScoring ?? DEFAULT_FANTASY_SCORING,
    projections: preseason,
    weekly: Object.fromEntries(
      Object.entries(payload.weekly ?? {}).map(([week, rows]) => [
        week,
        (rows ?? []).map((row) => normalizeProjection({ ...preseasonById.get(String((row as SupabaseFantasyRow).player_id)), ...row } as SupabaseFantasyRow)),
      ]),
    ),
    adp: payload.adp ?? [],
    metrics: payload.metrics ?? {},
    gaps: unique([...fallbackGaps, ...(payload.gaps ?? [])]),
    sources: payload.sources ?? [],
    dataSource: "static_json",
  };
}

export async function getFantasyFeed(scope: "preseason" | "week" = "preseason", week = 1): Promise<FantasyFeed> {
  const season = new Date().getUTCFullYear();
  const missing = getSupabaseMissingEnv();
  const resource = scope === "preseason" ? "fantasy_player_projections_latest" : "fantasy_player_projections_latest";
  // The public artifact currently contains more than 1,000 eligible players;
  // keep the live path from silently truncating the board while retaining a
  // bounded response for the browser.
  const query = `?select=*&season=eq.${season}&scope=eq.${scope}&week=eq.${scope === "week" ? week : 0}&order=points.desc&limit=5000`;
  const rows = await supabaseRest<SupabaseFantasyRow>(`${resource}${query}`);
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
      gaps: unique([...missing.map((item) => `Missing ${item}.`), `No fantasy artifact found at ${FANTASY_PATH}.`]),
      sources: [],
      dataSource: "unavailable",
    };
  }
}

export function scoreStatline(
  statline: Record<string, number>,
  scoring: FantasyScoring,
  position: FantasyPosition,
): number {
  const get = (key: string) => Number(statline[key] ?? 0);
  if (position === "K") {
    return round(
      get("extra_points_made") * scoring.kick_extra_point
        + get("fg_made_0_39") * scoring.field_goal_0_39
        + get("fg_made_40_49") * scoring.field_goal_40_49
        + get("fg_made_50_plus") * scoring.field_goal_50_plus
        + get("fg_missed") * scoring.missed_field_goal,
    );
  }
  if (position === "DST") {
    const pointsAllowed = get("dst_points_allowed");
    const points = pointsAllowed <= 0 ? scoring.dst_points_0
      : pointsAllowed <= 6 ? scoring.dst_points_1_6
        : pointsAllowed <= 13 ? scoring.dst_points_7_13
          : pointsAllowed <= 20 ? scoring.dst_points_14_20
            : pointsAllowed <= 27 ? scoring.dst_points_21_27
              : pointsAllowed <= 34 ? scoring.dst_points_28_34 : scoring.dst_points_35_plus;
    return round(
      points
        + get("dst_sacks") * scoring.dst_sack
        + get("dst_interceptions") * scoring.dst_interception
        + get("dst_fumble_recoveries") * scoring.dst_fumble_recovery
        + get("dst_tds") * scoring.dst_td
        + get("dst_safeties") * scoring.dst_safety
        + get("dst_blocked_kicks") * scoring.dst_blocked_kick,
    );
  }
  return round(
    get("passing_yards") * scoring.passing_yards
      + get("passing_tds") * scoring.passing_td
      + get("interceptions") * scoring.interception
      + get("rushing_yards") * scoring.rushing_yards
      + get("rushing_tds") * scoring.rushing_td
      + get("receptions") * scoring.reception
      + get("receiving_yards") * scoring.receiving_yards
      + get("receiving_tds") * scoring.receiving_td
      + get("fumbles_lost") * scoring.fumble_lost
      + get("two_point_conversions") * scoring.two_point_conversion,
  );
}

export function rescoreProjection(projection: FantasyProjection, scoring: FantasyScoring) {
  const median = scoreStatline(projection.statline, scoring, projection.position);
  const floor = scoreStatline(projection.statline_low ?? projection.statline, scoring, projection.position);
  const ceiling = scoreStatline(projection.statline_high ?? projection.statline, scoring, projection.position);
  return { median, floor, ceiling };
}

function round(value: number) {
  return Math.round(value * 100) / 100;
}
