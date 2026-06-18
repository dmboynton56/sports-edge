import { promises as fs } from "fs";
import path from "path";

export type ExperimentMetricSet = {
  modelVersion?: string | null;
  estimator?: string | null;
  status?: string | null;
  generatedAt?: string | null;
  trainingWindow?: Record<string, unknown> | null;
  featureColumns?: string[];
  categoricalColumns?: string[];
  leakageControls?: string[];
  pytorchWeight?: number | null;
  heuristicWeight?: number | null;
  validationLogLoss?: number | null;
  test?: {
    rows?: number | null;
    positiveRate?: number | null;
    brier?: number | null;
    baselineBrier?: number | null;
    logLoss?: number | null;
    baselineLogLoss?: number | null;
    auc?: number | null;
    top10HitRate?: number | null;
    top25HitRate?: number | null;
  } | null;
};

export type MlbHrExperimentSummary = {
  generatedAt: string | null;
  market: string;
  experimentStatus: string;
  baselineMetricsPath?: string;
  pytorchMetricsPath?: string;
  baseline: ExperimentMetricSet | null;
  pytorch: ExperimentMetricSet;
  pytorchBlend?: ExperimentMetricSet | null;
  pytorchHanded?: ExperimentMetricSet | null;
  pytorchHandedBlend?: ExperimentMetricSet | null;
  pytorchStatcast?: ExperimentMetricSet | null;
  pytorchStatcastBlend?: ExperimentMetricSet | null;
  dailyOutcomes?: {
    generatedAt?: string | null;
    status?: string | null;
    predictionRows?: number | null;
    evaluatedRows?: number | null;
    missingOutcomeRows?: number | null;
    evaluatedDates?: string[];
    modelVersions?: string[];
    modelProbability?: Record<string, number | string | null> | null;
    baselineProbability?: Record<string, number | string | null> | null;
  } | null;
  comparison: {
    brierDelta?: number | null;
    logLossDelta?: number | null;
    aucDelta?: number | null;
    top10HitRateDelta?: number | null;
    interpretation?: string;
  };
  blendComparison?: {
    brierDelta?: number | null;
    logLossDelta?: number | null;
    aucDelta?: number | null;
    top10HitRateDelta?: number | null;
    top25HitRateDelta?: number | null;
    interpretation?: string;
  };
  handedBlendComparison?: {
    brierDelta?: number | null;
    logLossDelta?: number | null;
    aucDelta?: number | null;
    top10HitRateDelta?: number | null;
    top25HitRateDelta?: number | null;
    interpretation?: string;
  };
  statcastBlendComparison?: {
    brierDelta?: number | null;
    logLossDelta?: number | null;
    aucDelta?: number | null;
    top10HitRateDelta?: number | null;
    top25HitRateDelta?: number | null;
    interpretation?: string;
  };
  dataExpansion: { name: string; url: string; use: string }[];
  blogDraft: {
    title: string;
    summary: string;
    publishWhen: string[];
  };
};

const EXPERIMENT_PATH = path.join(process.cwd(), "public", "data", "mlb_hr_experiment.json");

export async function getMlbHrExperimentSummary(): Promise<MlbHrExperimentSummary> {
  try {
    return JSON.parse(await fs.readFile(EXPERIMENT_PATH, "utf8")) as MlbHrExperimentSummary;
  } catch {
    return {
      generatedAt: null,
      market: "MLB batter home runs",
      experimentStatus: "missing_artifact",
      baseline: null,
      pytorch: {
        modelVersion: "mlb-hr-torch-v1",
        estimator: "pytorch_wide_deep",
        status: "pending_training",
        test: null,
      },
      pytorchBlend: null,
      pytorchHanded: null,
      pytorchHandedBlend: null,
      pytorchStatcast: null,
      pytorchStatcastBlend: null,
      dailyOutcomes: null,
      comparison: {
        interpretation: "Lower Brier/log loss and higher AUC/top-K hit rate are better.",
      },
      blendComparison: {
        interpretation: "Blend weights are selected on validation, then scored on the held-out test split.",
      },
      handedBlendComparison: {
        interpretation: "Handedness enrichment has not been evaluated yet.",
      },
      statcastBlendComparison: {
        interpretation: "Statcast enrichment has not been evaluated yet.",
      },
      dataExpansion: [],
      blogDraft: {
        title: "Can a GPU model find better MLB home-run probabilities?",
        summary: "Experiment summary artifact is missing.",
        publishWhen: [],
      },
    };
  }
}
