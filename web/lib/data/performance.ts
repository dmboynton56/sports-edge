import { promises as fs } from "fs";
import path from "path";

import type { Performance, PerformanceHistory } from "@/lib/data/types";

type RawPerformanceSport = {
  sport?: string;
  model_version?: string;
  modelVersion?: string;
  season?: string;
  market?: string;
  data_source?: string;
  dataSource?: string;
  sample?: Record<string, number | string | null>;
  metrics?: Record<string, string | number | null>;
  odds_status?: string;
  oddsStatus?: string;
  artifact_refs?: string[];
  artifactRefs?: string[];
  gaps?: string[];
  threshold_performance?: Record<string, string | number | null>[];
  thresholdPerformance?: Record<string, string | number | null>[];
  mode_performance?: Record<string, string | number | null>[];
  modePerformance?: Record<string, string | number | null>[];
};

type RawHistory =
  | {
      generated_at?: string;
      generatedAt?: string;
      oddspapi?: Record<string, string | number | null>;
      sports?: RawPerformanceSport[];
      records?: Performance[];
    }
  | undefined;

const PUBLIC_DATA_DIR = path.join(process.cwd(), "public", "data");
const PERFORMANCE_PATH = path.join(PUBLIC_DATA_DIR, "performance_history.json");

function numberMetric(
  metrics: Record<string, string | number | null> | undefined,
  names: string[],
) {
  for (const name of names) {
    const value = metrics?.[name];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function intMetric(
  metrics: Record<string, string | number | null> | undefined,
  names: string[],
) {
  const value = numberMetric(metrics, names);
  return value == null ? null : Math.trunc(value);
}

function sampleSize(sample: Record<string, number | string | null> | undefined) {
  if (!sample) return null;
  const preferred = [
    "completed_games",
    "test_games",
    "bigquery_scored_games",
    "test_rows_regression",
    "feature_rows",
    "supabase_graded_games",
  ];
  for (const key of preferred) {
    const value = sample[key];
    if (typeof value === "number") return value;
  }
  const first = Object.values(sample).find((value) => typeof value === "number");
  return typeof first === "number" ? first : null;
}

export function normalizePerformanceSport(raw: RawPerformanceSport): Performance {
  const metrics = raw.metrics ?? {};
  const wins = intMetric(metrics, ["supabase_ats_wins", "wins", "bigquery_default_wins"]);
  const losses = intMetric(metrics, ["supabase_ats_losses", "losses"]);
  const pushes = intMetric(metrics, ["supabase_ats_pushes", "pushes"]);
  const bets =
    intMetric(metrics, ["bigquery_default_bets", "bets", "n_bets"]) ??
    (wins != null && losses != null && pushes != null ? wins + losses + pushes : null);

  return {
    sport: raw.sport ?? "Unknown",
    modelVersion: raw.modelVersion ?? raw.model_version ?? "n/a",
    season: raw.season ?? "n/a",
    market: raw.market ?? "n/a",
    sampleSize: sampleSize(raw.sample),
    metrics: {
      ...metrics,
      accuracy: numberMetric(metrics, [
        "accuracy",
        "bigquery_accuracy",
        "bigquery_default_accuracy",
      ]),
      auc: numberMetric(metrics, ["auc", "roc_auc", "bigquery_auc", "win_auc"]),
      brier: numberMetric(metrics, ["brier", "bigquery_brier", "baseline_brier"]),
      logLoss: numberMetric(metrics, ["log_loss", "bigquery_log_loss", "baseline_log_loss"]),
      mae: numberMetric(metrics, ["mae", "bigquery_spread_mae", "sg_lgbm_mae"]),
      roi: numberMetric(metrics, [
        "supabase_ats_roi",
        "flat_roi",
        "bigquery_default_roi",
        "best_reported_sweep_roi",
      ]),
    },
    roi: numberMetric(metrics, [
      "supabase_ats_roi",
      "flat_roi",
      "bigquery_default_roi",
      "best_reported_sweep_roi",
    ]),
    units: numberMetric(metrics, ["units"]),
    bets,
    wins,
    losses,
    pushes,
    oddsStatus: raw.oddsStatus ?? raw.odds_status ?? "missing",
    dataSource: raw.dataSource ?? raw.data_source,
    sample: raw.sample ?? {},
    thresholdPerformance: raw.thresholdPerformance ?? raw.threshold_performance,
    modePerformance: raw.modePerformance ?? raw.mode_performance,
    artifactRefs: raw.artifactRefs ?? raw.artifact_refs ?? [],
    gaps: raw.gaps ?? [],
  };
}

export function normalizePerformanceHistory(raw: RawHistory): PerformanceHistory {
  const records =
    raw?.records ??
    (raw?.sports ?? []).map((sport) => normalizePerformanceSport(sport));
  return {
    generatedAt: raw?.generatedAt ?? raw?.generated_at ?? null,
    oddspapi: raw?.oddspapi,
    records,
    gaps: records.flatMap((record) => record.gaps.map((gap) => `${record.sport}: ${gap}`)),
  };
}

export async function getPerformanceHistory(): Promise<PerformanceHistory> {
  try {
    const raw = JSON.parse(await fs.readFile(PERFORMANCE_PATH, "utf8")) as RawHistory;
    return normalizePerformanceHistory(raw);
  } catch {
    return normalizePerformanceHistory(undefined);
  }
}
