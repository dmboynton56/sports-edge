import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(webRoot, "..");
const sourcePath = path.join(
  repoRoot,
  "data-core",
  "notebooks",
  "cache",
  "performance_history.json",
);
const outputPath = path.join(webRoot, "public", "data", "performance_history.json");

function metric(metrics, names) {
  for (const name of names) {
    const value = metrics?.[name];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function sampleSize(sample) {
  for (const key of [
    "completed_games",
    "test_games",
    "bigquery_scored_games",
    "test_rows_regression",
    "feature_rows",
    "supabase_graded_games",
  ]) {
    if (typeof sample?.[key] === "number") return sample[key];
  }
  return Object.values(sample ?? {}).find((value) => typeof value === "number") ?? null;
}

function gateStatus(gates) {
  if (gates.some((gate) => gate.status === "blocked")) return "blocked";
  if (gates.some((gate) => gate.status === "warning")) return "candidate";
  return "approved";
}

function metricMax(metrics, names) {
  const values = names
    .map((name) => metrics?.[name])
    .filter((value) => typeof value === "number" && Number.isFinite(value));
  return values.length ? Math.max(...values) : null;
}

function deriveProductionGates(sport) {
  const sportName = (sport.sport ?? "Unknown").toUpperCase();
  const metrics = sport.metrics ?? {};
  const sample = sampleSize(sport.sample);
  const oddsStatus = sport.odds_status ?? sport.oddsStatus ?? "missing";
  const bestRoi = metricMax(metrics, [
    "supabase_ats_roi",
    "flat_roi",
    "bigquery_default_roi",
    "best_reported_sweep_roi",
  ]);
  const hasCalibration =
    metric(metrics, ["brier", "bigquery_brier", "baseline_brier"]) != null ||
    metric(metrics, ["log_loss", "bigquery_log_loss", "baseline_log_loss"]) != null ||
    metric(metrics, ["auc", "roc_auc", "bigquery_auc", "win_auc"]) != null;
  const hasThresholds =
    (sport.threshold_performance ?? sport.thresholdPerformance ?? []).length > 0 ||
    (sport.mode_performance ?? sport.modePerformance ?? []).length > 0;
  const sampleTarget = sportName === "PGA" || sportName === "CBB" ? 500 : 100;
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
      status: sportName === "NFL" || sportName === "NBA" ? "warning" : "pass",
      detail:
        sportName === "NFL" || sportName === "NBA"
          ? "Injury schema and feature path exist; live impact coverage still needs rows."
          : "No active injury gate for this sport.",
    },
  ];
}

function normalizeSport(sport) {
  const metrics = sport.metrics ?? {};
  const wins = metric(metrics, ["supabase_ats_wins", "wins", "bigquery_default_wins"]);
  const losses = metric(metrics, ["supabase_ats_losses", "losses"]);
  const pushes = metric(metrics, ["supabase_ats_pushes", "pushes"]);
  const bets =
    metric(metrics, ["bigquery_default_bets", "bets", "n_bets"]) ??
    (wins != null && losses != null && pushes != null ? wins + losses + pushes : null);

  const productionGates =
    sport.production_gates ?? sport.productionGates ?? deriveProductionGates(sport);

  return {
    sport: sport.sport ?? "Unknown",
    modelVersion: sport.model_version ?? sport.modelVersion ?? "n/a",
    season: sport.season ?? "n/a",
    market: sport.market ?? "n/a",
    dataSource: sport.data_source ?? sport.dataSource ?? null,
    sampleSize: sampleSize(sport.sample),
    sample: sport.sample ?? {},
    metrics: {
      ...metrics,
      accuracy: metric(metrics, [
        "accuracy",
        "bigquery_accuracy",
        "bigquery_default_accuracy",
      ]),
      auc: metric(metrics, ["auc", "roc_auc", "bigquery_auc", "win_auc"]),
      brier: metric(metrics, ["brier", "bigquery_brier", "baseline_brier"]),
      logLoss: metric(metrics, ["log_loss", "bigquery_log_loss", "baseline_log_loss"]),
      mae: metric(metrics, ["mae", "bigquery_spread_mae", "sg_lgbm_mae"]),
      roi: metric(metrics, [
        "supabase_ats_roi",
        "flat_roi",
        "bigquery_default_roi",
        "best_reported_sweep_roi",
      ]),
    },
    roi: metric(metrics, [
      "supabase_ats_roi",
      "flat_roi",
      "bigquery_default_roi",
      "best_reported_sweep_roi",
    ]),
    units: metric(metrics, ["units"]),
    bets,
    wins,
    losses,
    pushes,
    oddsStatus: sport.odds_status ?? sport.oddsStatus ?? "missing",
    thresholdPerformance: sport.threshold_performance ?? sport.thresholdPerformance ?? [],
    modePerformance: sport.mode_performance ?? sport.modePerformance ?? [],
    artifactRefs: sport.artifact_refs ?? sport.artifactRefs ?? [],
    gaps: sport.gaps ?? [],
    productionGates,
    productionStatus: sport.production_status ?? sport.productionStatus ?? gateStatus(productionGates),
  };
}

const raw = JSON.parse(await readFile(sourcePath, "utf8"));
const records = (raw.sports ?? raw.records ?? []).map(normalizeSport);
const normalized = {
  generatedAt: raw.generated_at ?? raw.generatedAt ?? null,
  source: "data-core/notebooks/cache/performance_history.json",
  oddspapi: raw.oddspapi ?? null,
  records,
  gaps: records.flatMap((record) =>
    (record.gaps ?? []).map((gap) => `${record.sport}: ${gap}`),
  ),
};

await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(normalized, null, 2)}\n`);
console.log(`Wrote ${path.relative(repoRoot, outputPath)} (${records.length} records)`);
