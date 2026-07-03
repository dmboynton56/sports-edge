import { getSupabaseMissingEnv, getSupabaseRuntimeConfig } from "@/lib/data/supabase";

type GameResultRow = {
  league: string;
  season: number;
  week: number | null;
  game_date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  book_spread: number | null;
  my_spread: number | null;
  my_home_win_prob: number | null;
  model_version: string;
  spread_result: "win" | "loss" | "push" | null;
  winner_result: "win" | "loss" | null;
  flat_ats_units: number | null;
};

type MlbHrResultRow = {
  game_date: string;
  player_name: string;
  team: string | null;
  model_version: string;
  rank: number | null;
  top_k_bucket: string | null;
  model_probability: number | null;
  actual_home_run: boolean | null;
};

type PgaResultRow = {
  event_key: string;
  season: number | null;
  player_name: string;
  model_version: string;
  win_prob: number | null;
  top10_prob: number | null;
  top20_prob: number | null;
  final_position: string | null;
  final_position_numeric: number | null;
  top10_hit: boolean | null;
  top20_hit: boolean | null;
  winner_hit: boolean | null;
};

export type ResultsSummary = {
  league: string;
  market: string;
  modelVersion: string;
  sample: number;
  wins: number;
  losses: number;
  pushes: number;
  hitRate: number | null;
  roi: number | null;
};

export type ResultsData = {
  generatedAt: string;
  summaries: ResultsSummary[];
  gameRows: GameResultRow[];
  mlbHrRows: MlbHrResultRow[];
  pgaRows: PgaResultRow[];
  gaps: string[];
};

async function supabaseRest<T>(resource: string): Promise<T[] | null> {
  const config = getSupabaseRuntimeConfig();
  if (!config.url || !config.anonKey) return null;
  const base = config.url.replace(/\/$/, "");
  const response = await fetch(`${base}/rest/v1/${resource}`, {
    headers: {
      apikey: config.anonKey,
      Authorization: `Bearer ${config.anonKey}`,
    },
    next: { revalidate: 300 },
  });
  if (!response.ok) return null;
  return (await response.json()) as T[];
}

function rate(wins: number, losses: number) {
  const risked = wins + losses;
  return risked ? wins / risked : null;
}

function summarizeGameResults(rows: GameResultRow[]): ResultsSummary[] {
  const groups = new Map<string, GameResultRow[]>();
  for (const row of rows) {
    const key = `${row.league}|${row.model_version}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return Array.from(groups.entries()).flatMap(([key, group]) => {
    const [league, modelVersion] = key.split("|");
    const spreadWins = group.filter((row) => row.spread_result === "win").length;
    const spreadLosses = group.filter((row) => row.spread_result === "loss").length;
    const spreadPushes = group.filter((row) => row.spread_result === "push").length;
    const units = group.reduce((sum, row) => sum + (row.flat_ats_units ?? 0), 0);
    const winnerWins = group.filter((row) => row.winner_result === "win").length;
    const winnerLosses = group.filter((row) => row.winner_result === "loss").length;
    return [
      {
        league,
        market: "spread",
        modelVersion,
        sample: spreadWins + spreadLosses + spreadPushes,
        wins: spreadWins,
        losses: spreadLosses,
        pushes: spreadPushes,
        hitRate: rate(spreadWins, spreadLosses),
        roi: spreadWins + spreadLosses ? units / (spreadWins + spreadLosses) : null,
      },
      {
        league,
        market: "winner",
        modelVersion,
        sample: winnerWins + winnerLosses,
        wins: winnerWins,
        losses: winnerLosses,
        pushes: 0,
        hitRate: rate(winnerWins, winnerLosses),
        roi: null,
      },
    ];
  });
}

function summarizeMlbHr(rows: MlbHrResultRow[]): ResultsSummary[] {
  const groups = new Map<string, MlbHrResultRow[]>();
  for (const row of rows) {
    const key = `${row.model_version}|${row.top_k_bucket ?? "field"}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return Array.from(groups.entries()).map(([key, group]) => {
    const [modelVersion, bucket] = key.split("|");
    const wins = group.filter((row) => row.actual_home_run === true).length;
    const losses = group.filter((row) => row.actual_home_run === false).length;
    return {
      league: "MLB",
      market: `home_run ${bucket}`,
      modelVersion,
      sample: wins + losses,
      wins,
      losses,
      pushes: 0,
      hitRate: rate(wins, losses),
      roi: null,
    };
  });
}

function summarizePga(rows: PgaResultRow[]): ResultsSummary[] {
  const groups = new Map<string, PgaResultRow[]>();
  for (const row of rows) {
    groups.set(row.model_version, [...(groups.get(row.model_version) ?? []), row]);
  }
  return Array.from(groups.entries()).flatMap(([modelVersion, group]) => {
    const top10Wins = group.filter((row) => row.top10_hit === true).length;
    const top10Losses = group.filter((row) => row.top10_hit === false).length;
    const winnerWins = group.filter((row) => row.winner_hit === true).length;
    const winnerLosses = group.filter((row) => row.winner_hit === false).length;
    return [
      {
        league: "PGA",
        market: "top10",
        modelVersion,
        sample: top10Wins + top10Losses,
        wins: top10Wins,
        losses: top10Losses,
        pushes: 0,
        hitRate: rate(top10Wins, top10Losses),
        roi: null,
      },
      {
        league: "PGA",
        market: "winner",
        modelVersion,
        sample: winnerWins + winnerLosses,
        wins: winnerWins,
        losses: winnerLosses,
        pushes: 0,
        hitRate: rate(winnerWins, winnerLosses),
        roi: null,
      },
    ];
  });
}

export async function getResultsData(): Promise<ResultsData> {
  const [gameRows, mlbHrRows, pgaRows] = await Promise.all([
    supabaseRest<GameResultRow>("game_prediction_results?select=*&order=game_date.desc&limit=1000"),
    supabaseRest<MlbHrResultRow>("mlb_home_run_results?select=*&order=game_date.desc,rank.asc&limit=1000"),
    supabaseRest<PgaResultRow>("pga_prediction_results?select=*&order=evaluated_at.desc&limit=1000"),
  ]);

  const gaps = getSupabaseMissingEnv().map((name) => `Supabase results unavailable: missing ${name}.`);
  const summaries = [
    ...summarizeGameResults(gameRows ?? []),
    ...summarizeMlbHr(mlbHrRows ?? []),
    ...summarizePga(pgaRows ?? []),
  ].filter((row) => row.sample > 0);

  return {
    generatedAt: new Date().toISOString(),
    summaries,
    gameRows: gameRows ?? [],
    mlbHrRows: mlbHrRows ?? [],
    pgaRows: pgaRows ?? [],
    gaps,
  };
}
