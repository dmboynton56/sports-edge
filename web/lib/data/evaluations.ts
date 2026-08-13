import { getSupabaseMissingEnv, supabaseRest } from "@/lib/data/supabase";

export type EvaluationMetrics = {
  accuracy: number | null;
  auc: number | null;
  brier: number | null;
  logLoss: number | null;
  roi: number | null;
};

export type EvaluationRow = {
  league: string;
  model_name: string;
  model_version: string;
  evaluation_name: string;
  test_start_date: string | null;
  test_end_date: string | null;
  generated_at: string;
  metrics: Record<string, unknown>;
  status: string;
  displayMetrics: EvaluationMetrics;
};

export type EvaluationData = {
  rows: EvaluationRow[];
  gaps: string[];
};

type RawEvaluationRow = Omit<EvaluationRow, "displayMetrics">;

function numberMetric(metrics: Record<string, unknown>, names: string[]) {
  for (const name of names) {
    const value = metrics[name];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function normalizeRow(row: RawEvaluationRow): EvaluationRow {
  const metrics = row.metrics && typeof row.metrics === "object" ? row.metrics : {};
  return {
    ...row,
    metrics,
    displayMetrics: {
      accuracy: numberMetric(metrics, ["accuracy", "bigquery_accuracy", "bigquery_default_accuracy"]),
      auc: numberMetric(metrics, ["auc", "roc_auc", "bigquery_auc", "win_auc"]),
      brier: numberMetric(metrics, ["brier", "bigquery_brier", "baseline_brier"]),
      logLoss: numberMetric(metrics, ["log_loss", "bigquery_log_loss", "baseline_log_loss"]),
      roi: numberMetric(metrics, [
        "supabase_ats_roi",
        "flat_roi",
        "bigquery_default_roi",
        "best_reported_sweep_roi",
      ]),
    },
  };
}

function missingEnvGaps(source: string) {
  return getSupabaseMissingEnv().map(
    (name) => `${source} unavailable: missing ${name}.`,
  );
}

async function getEvaluations(
  table: "model_evaluation_runs" | "model_evaluation_history",
  league?: string,
): Promise<EvaluationData> {
  const missing = missingEnvGaps(
    table === "model_evaluation_runs" ? "Model evaluations" : "Evaluation history",
  );
  if (missing.length) return { rows: [], gaps: missing };

  const leagueFilter = league ? `&league=eq.${encodeURIComponent(league.toUpperCase())}` : "";
  const rows = await supabaseRest<RawEvaluationRow>(
    `${table}?select=league,model_name,model_version,evaluation_name,test_start_date,test_end_date,generated_at,metrics,status${leagueFilter}&order=generated_at.desc&limit=200`,
  );
  if (rows == null) {
    return {
      rows: [],
      gaps: [
        table === "model_evaluation_history"
          ? "model_evaluation_history not available"
          : "model_evaluation_runs not available",
      ],
    };
  }
  return { rows: rows.map(normalizeRow), gaps: [] };
}

export function getEvaluationRuns(league?: string) {
  return getEvaluations("model_evaluation_runs", league);
}

export function getEvaluationHistory(league?: string) {
  return getEvaluations("model_evaluation_history", league);
}

import type { Performance } from "@/lib/data/types";
import { getPerformanceHistory } from "@/lib/data/performance";
import { getSupabaseRuntimeConfig } from "@/lib/data/supabase";

export type ModelEvaluation = {
  id: string;
  league: string;
  modelName: string;
  modelVersion: string;
  evaluationName: string;
  generatedAt: string;
  status: string;
  metrics: Record<string, unknown>;
  calibration: Record<string, unknown>;
  artifactRefs: string[];
  notes: string | null;
};

export type StrategyBacktest = {
  id: string;
  league: string;
  modelName: string;
  modelVersion: string;
  strategyId: string;
  market: string;
  oddsSource: string | null;
  sampleSize: number | null;
  bets: number | null;
  roi: number | null;
  metrics: Record<string, unknown>;
};

export type ModelRegistryEntry = {
  league: string;
  modelVersion: string;
  status: "production" | "candidate" | "archived";
  notes: string;
};

const REGISTRY: ModelRegistryEntry[] = [
  { league: "NBA", modelVersion: "v3", status: "production", notes: "Daily refresh spread + win prob" },
  { league: "NFL", modelVersion: "v1", status: "production", notes: "Weekly spread + win prob" },
  { league: "MLB", modelVersion: "v3", status: "production", notes: "Probability-only display" },
];

type SupabaseEvalRow = {
  id: string;
  league: string;
  model_name: string;
  model_version: string;
  evaluation_name: string;
  generated_at: string;
  status: string;
  metrics: Record<string, unknown>;
  calibration: Record<string, unknown>;
  artifact_refs: string[];
  notes: string | null;
};

type SupabaseStrategyRow = {
  id: string;
  league: string;
  model_name: string;
  model_version: string;
  strategy_id: string;
  market: string;
  odds_source: string | null;
  sample_size: number | null;
  bets: number | null;
  roi: number | null;
  metrics: Record<string, unknown>;
};

async function evaluationSupabaseRest<T>(resource: string): Promise<T[] | null> {
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

function mapEval(row: SupabaseEvalRow): ModelEvaluation {
  return {
    id: row.id,
    league: row.league,
    modelName: row.model_name,
    modelVersion: row.model_version,
    evaluationName: row.evaluation_name,
    generatedAt: row.generated_at,
    status: row.status,
    metrics: row.metrics ?? {},
    calibration: row.calibration ?? {},
    artifactRefs: row.artifact_refs ?? [],
    notes: row.notes,
  };
}

function mapStrategy(row: SupabaseStrategyRow): StrategyBacktest {
  return {
    id: row.id,
    league: row.league,
    modelName: row.model_name,
    modelVersion: row.model_version,
    strategyId: row.strategy_id,
    market: row.market,
    oddsSource: row.odds_source,
    sampleSize: row.sample_size,
    bets: row.bets,
    roi: row.roi,
    metrics: row.metrics ?? {},
  };
}

function performanceToEval(record: Performance): ModelEvaluation {
  return {
    id: `${record.sport}-${record.modelVersion}`,
    league: record.sport,
    modelName: "sports_edge",
    modelVersion: record.modelVersion,
    evaluationName: `${record.market}-${record.season}`,
    generatedAt: new Date().toISOString(),
    status: record.productionStatus,
    metrics: record.metrics,
    calibration: {},
    artifactRefs: record.artifactRefs,
    notes: record.gaps.join("; ") || null,
  };
}

function performanceToStrategy(record: Performance): StrategyBacktest | null {
  if (record.roi == null) return null;
  return {
    id: `${record.sport}-${record.modelVersion}-strategy`,
    league: record.sport,
    modelName: "sports_edge",
    modelVersion: record.modelVersion,
    strategyId: `${record.market}-flat`,
    market: record.market,
    oddsSource: record.dataSource ?? null,
    sampleSize: record.sampleSize,
    bets: record.bets,
    roi: record.roi,
    metrics: record.metrics,
  };
}

export async function getModelEvaluations(): Promise<ModelEvaluation[]> {
  const rows = await evaluationSupabaseRest<SupabaseEvalRow>(
    "model_evaluation_runs?order=generated_at.desc&limit=100",
  );
  if (rows?.length) return rows.map(mapEval);

  const history = await getPerformanceHistory();
  return history.records.map(performanceToEval);
}

export async function getStrategyBacktests(): Promise<StrategyBacktest[]> {
  const rows = await evaluationSupabaseRest<SupabaseStrategyRow>(
    "strategy_backtest_results?order=created_at.desc&limit=100",
  );
  if (rows?.length) return rows.map(mapStrategy);

  const history = await getPerformanceHistory();
  return history.records
    .map(performanceToStrategy)
    .filter((row): row is StrategyBacktest => row != null);
}

export function getModelRegistry(): ModelRegistryEntry[] {
  return REGISTRY;
}

export async function getEvaluationsBundle() {
  const [evaluations, strategies, registry] = await Promise.all([
    getModelEvaluations(),
    getStrategyBacktests(),
    Promise.resolve(getModelRegistry()),
  ]);
  const gaps = getSupabaseMissingEnv();
  return {
    generatedAt: new Date().toISOString(),
    evaluations,
    strategies,
    registry,
    gaps: gaps.length ? [`Supabase eval tables unavailable: ${gaps.join(", ")}`] : [],
  };
}
