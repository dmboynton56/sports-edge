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
