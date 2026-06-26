import type { Prediction } from "@/lib/data/types";

export type MlbHomeRunPrediction = Prediction & {
  gameDate?: string | null;
  team?: string | null;
  opponent?: string | null;
  venue?: string | null;
  lineupSlot?: number | null;
  lineupStatus?: string | null;
  opposingProbablePitcher?: string | null;
  baselineProbability?: number | null;
  gamesSinceLastHr?: number | null;
  lastHrDate?: string | null;
  rank?: number | null;
  qualityFlags?: string[];
  topFeatures?: { feature: string; value: number }[];
  bestBook?: string | null;
  bestBookTitle?: string | null;
  bestPrice?: number | null;
  noVigProbability?: number | null;
  marketProbability?: number | null;
  oddsBooksCount?: number | null;
  oddsSnapshotTs?: string | null;
  oddsStatus?: string | null;
};

export type MlbHomeRunModelFeed = {
  modelVersion: string;
  predictions: MlbHomeRunPrediction[];
  gaps: string[];
};

export type MlbHomeRunFeed = {
  generatedAt: string | null;
  modelVersion: string;
  defaultModel?: string;
  productionStatus: "candidate" | "approved" | "blocked";
  predictions: MlbHomeRunPrediction[];
  gaps: string[];
  models?: Record<string, MlbHomeRunModelFeed>;
};

export type MlbHomeRunBoardData = {
  generatedAt: string | null;
  productionStatus: MlbHomeRunFeed["productionStatus"];
  defaultModel: string;
  availableModels: string[];
  models: Record<string, MlbHomeRunModelFeed>;
};

export const MLB_HR_V1_MODEL = "mlb-hr-v1";
export const MLB_HR_STATCAST_BLEND_MODEL = "mlb-hr-torch-statcast-v1-blend";

const MLB_MODEL_LABELS: Record<string, string> = {
  [MLB_HR_V1_MODEL]: "Random Forest (v1)",
  [MLB_HR_STATCAST_BLEND_MODEL]: "Statcast blend",
};

export function getMlbHomeRunModelLabel(modelKey: string): string {
  return MLB_MODEL_LABELS[modelKey] ?? modelKey;
}
