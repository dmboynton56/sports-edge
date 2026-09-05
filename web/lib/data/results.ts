import { getSupabaseMissingEnv, supabaseRest } from "@/lib/data/supabase";
import { isFiniteNumber, isJsonString } from "@/lib/data/json";

export type GameResultRow = {
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
  evaluated_at?: string | null;
};

export type MlbHrResultRow = {
  board_row_id: string | null;
  game_date: string;
  game_id?: string;
  player_name: string;
  team: string | null;
  model_version: string;
  rank: number | null;
  top_k_bucket: string | null;
  model_probability: number | null;
  actual_home_run: boolean | null;
  actual_home_runs?: number | null;
  actual_plate_appearances?: number | null;
  evaluated_at?: string | null;
  american_price?: number | null;
  market_probability?: number | null;
  edge?: number | null;
  ev?: number | null;
  odds_status?: string | null;
};

export type PgaResultRow = {
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
  evaluated_at: string;
};

export type ResultsWindow = "7d" | "30d" | "season" | "all";

export type WeeklyResultsBucket = {
  weekStart: string;
  wins: number;
  losses: number;
  pushes: number;
  hitRate: number | null;
  units: number;
};

export type RawResults<T> = {
  rows: T[];
  gaps: string[];
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
  pricedSample?: number;
  pricedHitRate?: number | null;
  flatUnits?: number | null;
  modelOnlySample?: number;
  modelOnlyHitRate?: number | null;
};

export type ResultsData = {
  generatedAt: string | null;
  summaries: ResultsSummary[];
  gameRows: GameResultRow[];
  mlbHrRows: MlbHrResultRow[];
  pgaRows: PgaResultRow[];
  gaps: string[];
};

// Exclude unused raw records and publication metadata to reduce transfer size
// and keep the current results response within the Next.js fetch cache limit.
const MLB_HR_RESULTS_QUERY = "mlb_home_run_published_results?select=" + [
  "board_row_id", "game_date", "game_id", "player_name", "team", "model_version",
  "rank", "top_k_bucket", "model_probability", "actual_home_run", "actual_home_runs",
  "actual_plate_appearances", "evaluated_at", "american_price", "market_probability",
  "edge", "ev", "odds_status",
].join(",") + "&order=game_date.desc,rank.asc&limit=5000";

function rate(wins: number, losses: number) {
  const risked = wins + losses;
  return risked ? wins / risked : null;
}

export function summarizeGameResults(rows: GameResultRow[]): ResultsSummary[] {
  const groups = new Map<string, GameResultRow[]>();
  for (const row of rows) {
    const key = `${row.league}|${row.model_version}`;
    const group = groups.get(key);
    if (group) group.push(row);
    else groups.set(key, [row]);
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

export function summarizeMlbHr(rows: MlbHrResultRow[]): ResultsSummary[] {
  const groups = new Map<string, MlbHrResultRow[]>();
  for (const row of rows) {
    const key = `${row.model_version}|${row.top_k_bucket ?? "field"}`;
    const group = groups.get(key);
    if (group) group.push(row);
    else groups.set(key, [row]);
  }
  return Array.from(groups.entries()).map(([key, group]) => {
    const [modelVersion, bucket] = key.split("|");
    const evaluable = group.filter((row) => (row.actual_plate_appearances ?? 0) > 0 && row.actual_home_run != null);
    const wins = evaluable.filter((row) => row.actual_home_run === true).length;
    const losses = evaluable.filter((row) => row.actual_home_run === false).length;
    const priced = evaluable.filter(
      (row) =>
        isFiniteNumber(row.american_price) &&
        row.american_price !== 0 &&
        (row.odds_status === "ok" || row.odds_status === "raw_implied"),
    );
    const pricedWins = priced.filter((row) => row.actual_home_run === true).length;
    const pricedSet = new Set(priced);
    const modelOnly = evaluable.filter((row) => !pricedSet.has(row));
    const modelOnlyWins = modelOnly.filter((row) => row.actual_home_run === true).length;
    const flatUnits = priced.reduce((sum, row) => {
      if (row.actual_home_run !== true && row.actual_home_run !== false) return sum;
      const price = row.american_price ?? 0;
      const profit = row.actual_home_run
        ? price > 0
          ? price / 100
          : 100 / Math.abs(price)
        : -1;
      return sum + profit;
    }, 0);
    return {
      league: "MLB",
      market: `home_run ${bucket}`,
      modelVersion,
      sample: wins + losses,
      wins,
      losses,
      pushes: 0,
      hitRate: rate(wins, losses),
      roi: priced.length ? flatUnits / priced.length : null,
      pricedSample: priced.length,
      pricedHitRate: rate(pricedWins, priced.length - pricedWins),
      flatUnits,
      modelOnlySample: modelOnly.length,
      modelOnlyHitRate: rate(modelOnlyWins, modelOnly.length - modelOnlyWins),
    };
  });
}

export function summarizePga(rows: PgaResultRow[]): ResultsSummary[] {
  const groups = new Map<string, PgaResultRow[]>();
  for (const row of rows) {
    const group = groups.get(row.model_version);
    if (group) group.push(row);
    else groups.set(row.model_version, [row]);
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

function resultGaps(source: string, rows: unknown[] | null) {
  const missing = getSupabaseMissingEnv();
  if (missing.length) {
    return missing.map((name) => `Supabase ${source} unavailable: missing ${name}.`);
  }
  return rows == null ? [`Supabase ${source} unavailable.`] : [];
}

export function filterByWindow<T>(
  rows: T[],
  window: ResultsWindow,
  dateField: keyof T,
  now = new Date(),
): T[] {
  if (window === "all") return rows;

  const cutoff = new Date(now);
  if (window === "season") {
    cutoff.setUTCMonth(0, 1);
    cutoff.setUTCHours(0, 0, 0, 0);
  } else {
    cutoff.setUTCDate(cutoff.getUTCDate() - (window === "7d" ? 7 : 30));
  }

  return rows.filter((row) => {
    const value = row[dateField];
    if (!isJsonString(value)) return false;
    const date = new Date(value.length === 10 ? `${value}T00:00:00Z` : value);
    return !Number.isNaN(date.getTime()) && date >= cutoff;
  });
}

function mondayStart(value: string) {
  const date = new Date(value.length === 10 ? `${value}T00:00:00Z` : value);
  if (Number.isNaN(date.getTime())) return null;
  date.setUTCHours(0, 0, 0, 0);
  const day = date.getUTCDay();
  date.setUTCDate(date.getUTCDate() - (day === 0 ? 6 : day - 1));
  return date.toISOString().slice(0, 10);
}

export function bucketWeeklyResults(
  rows: GameResultRow[],
  resultField: "spread_result" | "winner_result" = "spread_result",
): WeeklyResultsBucket[] {
  const buckets = new Map<string, WeeklyResultsBucket>();
  for (const row of rows) {
    const weekStart = mondayStart(row.game_date);
    const result = row[resultField];
    if (!weekStart || !result) continue;
    const bucket = buckets.get(weekStart) ?? {
      weekStart,
      wins: 0,
      losses: 0,
      pushes: 0,
      hitRate: null,
      units: 0,
    };
    if (result === "win") bucket.wins += 1;
    if (result === "loss") bucket.losses += 1;
    if (result === "push") bucket.pushes += 1;
    if (resultField === "spread_result") bucket.units += row.flat_ats_units ?? 0;
    buckets.set(weekStart, bucket);
  }

  return Array.from(buckets.values())
    .map((bucket) => ({
      ...bucket,
      hitRate: rate(bucket.wins, bucket.losses),
    }))
    .sort((a, b) => a.weekStart.localeCompare(b.weekStart));
}

export async function getGameResultRows(league: string): Promise<RawResults<GameResultRow>> {
  const normalizedLeague = league.toUpperCase();
  const rows = await supabaseRest<GameResultRow>(
    `game_prediction_results?select=*&league=eq.${encodeURIComponent(normalizedLeague)}&order=game_date.desc&limit=5000`,
  );
  return { rows: rows ?? [], gaps: resultGaps(`${normalizedLeague} game results`, rows) };
}

export async function getMlbHomeRunResultRows(): Promise<RawResults<MlbHrResultRow>> {
  const rows = await supabaseRest<MlbHrResultRow>(
    MLB_HR_RESULTS_QUERY,
  );
  return { rows: rows ?? [], gaps: resultGaps("MLB home run results", rows) };
}

export async function getPgaResultRows(): Promise<RawResults<PgaResultRow>> {
  const rows = await supabaseRest<PgaResultRow>(
    "pga_prediction_results?select=*&order=evaluated_at.desc&limit=5000",
  );
  return { rows: rows ?? [], gaps: resultGaps("PGA results", rows) };
}

export async function getResultsData(): Promise<ResultsData> {
  const [gameRows, mlbHrRows, pgaRows] = await Promise.all([
    supabaseRest<GameResultRow>("game_prediction_results?select=*&order=game_date.desc&limit=5000"),
    supabaseRest<MlbHrResultRow>(MLB_HR_RESULTS_QUERY),
    supabaseRest<PgaResultRow>("pga_prediction_results?select=*&order=evaluated_at.desc&limit=5000"),
  ]);

  const gaps = [
    ...getSupabaseMissingEnv().map((name) => `Supabase results unavailable: missing ${name}.`),
    ...resultGaps("game results", gameRows),
    ...resultGaps("MLB home run published results", mlbHrRows),
    ...resultGaps("PGA results", pgaRows),
  ];
  const summaries = [
    ...summarizeGameResults(gameRows ?? []),
    ...summarizeMlbHr(mlbHrRows ?? []),
    ...summarizePga(pgaRows ?? []),
  ].filter((row) => row.sample > 0);

  const timestamps = [
    ...(gameRows ?? []).map((row) => row.evaluated_at),
    ...(mlbHrRows ?? []).map((row) => row.evaluated_at),
    ...(pgaRows ?? []).map((row) => row.evaluated_at),
  ].filter((value): value is string => Boolean(value));
  return {
    generatedAt: timestamps.toSorted().at(-1) ?? null,
    summaries,
    gameRows: gameRows ?? [],
    mlbHrRows: mlbHrRows ?? [],
    pgaRows: pgaRows ?? [],
    gaps,
  };
}
