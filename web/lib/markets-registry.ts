export type MarketEntry = {
  slug: string;
  label: string;
  /** Two words at most — used in chips and dense lists. */
  short: string;
  href: string;
  status: "live" | "scaffold";
  description: string;
};

export type SportEntry = {
  slug: "nba" | "mlb" | "pga" | "nfl" | "nhl" | "cfb" | "cbb";
  label: string;
  emphasis: "primary" | "scaffold" | "seasonal";
  description: string;
  markets: MarketEntry[];
};

export const SPORTS: SportEntry[] = [
  {
    slug: "nba",
    label: "NBA",
    emphasis: "primary",
    description: "Team spread and winner probabilities.",
    markets: [
      {
        slug: "spread-winner",
        label: "Spread & winner board",
        short: "spread",
        href: "/markets/nba",
        status: "live",
        description: "Pre-live team model probabilities and market edges.",
      },
    ],
  },
  {
    slug: "mlb",
    label: "MLB",
    emphasis: "primary",
    description: "Daily player and team baseball markets.",
    markets: [
      {
        slug: "home-runs",
        label: "Home runs",
        short: "home runs",
        href: "/markets/mlb/home-runs",
        status: "live",
        description: "Daily batter home-run probabilities and market edges.",
      },
      {
        slug: "winners",
        label: "Winners",
        short: "winners",
        href: "/markets/mlb",
        status: "live",
        description: "Pre-live team winner probabilities.",
      },
      {
        slug: "run-line",
        label: "Run line",
        short: "run line",
        href: "/markets/mlb/run-line",
        status: "live",
        description: "Run-line margin-residual research model (v1). Model-only when sportsbook prices are unavailable.",
      },
      {
        slug: "totals",
        label: "Totals",
        short: "totals",
        href: "/markets/mlb/totals",
        status: "live",
        description: "Over/under research model (v1). Model-only when sportsbook prices are unavailable.",
      },
      {
        slug: "pitcher-strikeouts",
        label: "Pitcher strikeouts",
        short: "pitcher Ks",
        href: "/markets/mlb",
        status: "scaffold",
        description: "Starter strikeout research model (v1). Artifact not committed; codex will retrain.",
      },
    ],
  },
  {
    slug: "pga",
    label: "PGA",
    emphasis: "primary",
    description: "Tournament placement and winner probabilities.",
    markets: [
      {
        slug: "tournament-board",
        label: "Tournament board — win/top-10/top-20",
        short: "placements",
        href: "/markets/pga",
        status: "live",
        description: "Tournament outlook, placement markets, and recent form.",
      },
    ],
  },
  {
    slug: "nfl",
    label: "NFL",
    emphasis: "primary",
    description: "Team markets and anytime-touchdown probabilities.",
    markets: [
      {
        slug: "spread-winner",
        label: "Week 1 team & TD board",
        short: "Week 1",
        href: "/markets/nfl",
        status: "live",
        description: "Moneyline, spread, total, and guarded anytime-touchdown research signals.",
      },
    ],
  },
  {
    slug: "cfb",
    label: "CFB",
    emphasis: "primary",
    description: "Daily college-football team markets.",
    markets: [
      {
        slug: "team-markets",
        label: "Points, winner & lines",
        short: "team markets",
        href: "/markets/cfb",
        status: "live",
        description: "Projected scores, win chances, moneylines, spreads, and totals.",
      },
    ],
  },
  {
    slug: "nhl",
    label: "NHL",
    emphasis: "scaffold",
    description: "Hockey market integration scaffold.",
    markets: [
      {
        slug: "models",
        label: "No models yet",
        short: "not started",
        href: "/markets/nhl",
        status: "scaffold",
        description: "No NHL models are wired to the dashboard.",
      },
    ],
  },
  {
    slug: "cbb",
    label: "CBB",
    emphasis: "seasonal",
    description: "Seasonal college basketball tournament modeling.",
    markets: [
      {
        slug: "tournament-bracket",
        label: "Tournament bracket sim",
        short: "bracket",
        href: "/markets/cbb",
        status: "live",
        description: "Interactive bracket and Monte Carlo simulation.",
      },
    ],
  },
];

export function getSport(slug: string): SportEntry | undefined {
  return SPORTS.find((sport) => sport.slug === slug);
}
