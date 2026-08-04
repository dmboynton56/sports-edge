import type { SportEntry } from "@/lib/markets-registry";

export type SportSlug = SportEntry["slug"];

/**
 * Every league carries one color across the whole site — chips, table swatches,
 * chart series. Class names are spelled out because Tailwind scans source text.
 */
const SPORT_CLASSES: Record<SportSlug, { fill: string; text: string }> = {
  nba: { fill: "bg-nba", text: "text-nba" },
  mlb: { fill: "bg-mlb", text: "text-mlb" },
  pga: { fill: "bg-pga", text: "text-pga" },
  nfl: { fill: "bg-nfl", text: "text-nfl" },
  nhl: { fill: "bg-nhl", text: "text-nhl" },
  cbb: { fill: "bg-cbb", text: "text-cbb" },
};

const FALLBACK = { fill: "bg-muted-foreground", text: "text-muted-foreground" };

export function sportColor(slug: string | null | undefined) {
  if (!slug) return FALLBACK;
  return SPORT_CLASSES[slug.toLowerCase() as SportSlug] ?? FALLBACK;
}

/** Charts need a raw color, not a class. Matches the tokens in globals.css. */
export function sportVar(slug: string | null | undefined) {
  const key = slug?.toLowerCase() ?? "";
  return key in SPORT_CLASSES ? `hsl(var(--sport-${key}))` : "hsl(var(--muted-foreground))";
}
