import { promises as fs } from "fs";
import path from "path";

import type { Performance, PerformanceHistory, ProductionGate } from "@/lib/data/types";

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
  production_status?: Performance["productionStatus"];
  productionStatus?: Performance["productionStatus"];
  production_gates?: ProductionGate[];
  productionGates?: ProductionGate[];
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

function gateStatus(gates: ProductionGate[]): Performance["productionStatus"] {
  if (gates.some((gate) => gate.status === "blocked")) return "blocked";
  if (gates.some((gate) => gate.status === "warning")) return "candidate";
  return "approved";
}

function metricMax(metrics: Record<string, string | number | null> | undefined, names: string[]) {
  const values = names
    .map((name) => metrics?.[name])
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return values.length ? Math.max(...values) : null;
}

function deriveProductionGates(raw: RawPerformanceSport): ProductionGate[] {
  const sport = (raw.sport ?? "Unknown").toUpperCase();
  const metrics = raw.metrics ?? {};
  const sample = sampleSize(raw.sample);
  const oddsStatus = raw.oddsStatus ?? raw.odds_status ?? "missing";
  const bestRoi = metricMax(metrics, [
    "supabase_ats_roi",
    "flat_roi",
    "bigquery_default_roi",
    "best_reported_sweep_roi",
  ]);
  const hasCalibration =
    numberMetric(metrics, ["brier", "bigquery_brier", "baseline_brier"]) != null ||
    numberMetric(metrics, ["log_loss", "bigquery_log_loss", "baseline_log_loss"]) != null ||
    numberMetric(metrics, ["auc", "roc_auc", "bigquery_auc", "win_auc"]) != null;
  const hasThresholds =
    (raw.thresholdPerformance ?? raw.threshold_performance ?? []).length > 0 ||
    (raw.modePerformance ?? raw.mode_performance ?? []).length > 0;
  const sampleTarget = sport === "PGA" || sport === "CBB" ? 500 : 100;
  const oddsLower = oddsStatus.toLowerCase();

  return [
    {
      id: "sample",
      label: "Sample",
      status: sample != null && sample >= sampleTarget ? "pass" : "warning",
      detail:
        sample == null
          ? "No sample size recorded."
          : `${sample.toLocaleString("en-US")} rows/games recorded.`,
    },
    {
      id: "calibration",
      label: "Calibration",
      status: hasCalibration ? "pass" : "warning",
      detail: hasCalibration ? "Calibration metrics are recorded." : "Needs Brier, log loss, or AUC evidence.",
    },
    {
      id: "strategy",
      label: "Strategy ROI",
      status: bestRoi == null ? "warning" : bestRoi > 0 ? "pass" : "blocked",
      detail:
        bestRoi == null
          ? "No strategy ROI recorded."
          : `Best recorded ROI ${(bestRoi * 100).toFixed(1)}%.`,
    },
    {
      id: "odds",
      label: "Odds",
      status:
        oddsLower.includes("missing") || oddsLower.includes("no_sportsbook")
          ? "blocked"
          : oddsLower.includes("partial") || oddsLower.includes("free")
            ? "warning"
            : "pass",
      detail: oddsStatus,
    },
    {
      id: "thresholds",
      label: "Thresholds",
      status: hasThresholds ? "pass" : "warning",
      detail: hasThresholds ? "Threshold or mode sweeps are available." : "No threshold sweep evidence recorded.",
    },
    {
      id: "injuries",
      label: "Injuries",
      status: sport === "NFL" || sport === "NBA" ? "warning" : "pass",
      detail:
        sport === "NFL" || sport === "NBA"
          ? "Injury schema and feature path exist; live impact coverage still needs rows."
          : "No active injury gate for this sport.",
    },
  ];
}

export function normalizePerformanceSport(raw: RawPerformanceSport): Performance {
  const metrics = raw.metrics ?? {};
  const wins = intMetric(metrics, ["supabase_ats_wins", "wins", "bigquery_default_wins"]);
  const losses = intMetric(metrics, ["supabase_ats_losses", "losses"]);
  const pushes = intMetric(metrics, ["supabase_ats_pushes", "pushes"]);
  const bets =
    intMetric(metrics, ["bigquery_default_bets", "bets", "n_bets"]) ??
    (wins != null && losses != null && pushes != null ? wins + losses + pushes : null);

  const productionGates = raw.productionGates ?? raw.production_gates ?? deriveProductionGates(raw);

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
    productionGates,
    productionStatus: raw.productionStatus ?? raw.production_status ?? gateStatus(productionGates),
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
