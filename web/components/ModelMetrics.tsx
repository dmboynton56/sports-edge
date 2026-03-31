'use client';

import React, { useState } from 'react';

interface RegressionRow {
  model: string;
  rmse: number;
  mae: number;
  spearman: number;
  ndcg10: number;
  ndcg20: number;
}

interface ClassificationRow {
  model: string;
  brier: number;
  logLoss: number;
  auc: number;
  ece: number;
}

interface ClassificationTarget {
  name: string;
  baseRate: number;
  rows: ClassificationRow[];
}

interface FeatureImportance {
  name: string;
  importance: number;
}

export interface ModelMetricsData {
  regression: RegressionRow[];
  classification: ClassificationTarget[];
  topFeatures: FeatureImportance[];
  metaWeights: { model: string; weight: number }[];
}

function MetricCell({ value, best, format = 'f4', lowerBetter = true }: {
  value: number; best: number; format?: string; lowerBetter?: boolean;
}) {
  const isBest = Math.abs(value - best) < 0.0001;
  const formatted = format === 'pct' ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
  return (
    <td className={`px-3 py-2.5 text-right font-mono text-sm ${isBest ? 'text-emerald-400 font-bold' : 'text-muted-foreground'}`}>
      {formatted}
    </td>
  );
}

export function ModelMetrics({ data }: { data: ModelMetricsData }) {
  const [activeClsTab, setActiveClsTab] = useState(0);

  const bestRMSE = Math.min(...data.regression.map(r => r.rmse));
  const bestMAE = Math.min(...data.regression.map(r => r.mae));
  const bestSpearman = Math.max(...data.regression.map(r => r.spearman));
  const bestNDCG10 = Math.max(...data.regression.map(r => r.ndcg10));
  const bestNDCG20 = Math.max(...data.regression.map(r => r.ndcg20));

  const activeTarget = data.classification[activeClsTab];
  const bestBrier = Math.min(...activeTarget.rows.map(r => r.brier));
  const bestLL = Math.min(...activeTarget.rows.map(r => r.logLoss));
  const bestAUC = Math.max(...activeTarget.rows.map(r => r.auc));
  const bestECE = Math.min(...activeTarget.rows.map(r => r.ece));

  return (
    <div className="space-y-6">
      {/* Regression */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="px-5 py-4 border-b border-border bg-secondary/20">
          <h3 className="font-semibold text-sm">Regression Models — Strokes Gained Prediction</h3>
          <p className="text-xs text-muted-foreground mt-1">Validated on 3,071 holdout tournament entries (2022 season)</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left px-3 py-2.5 font-medium text-muted-foreground">Model</th>
                <th className="text-right px-3 py-2.5 font-medium text-muted-foreground">RMSE</th>
                <th className="text-right px-3 py-2.5 font-medium text-muted-foreground">MAE</th>
                <th className="text-right px-3 py-2.5 font-medium text-muted-foreground">Spearman</th>
                <th className="text-right px-3 py-2.5 font-medium text-muted-foreground">NDCG@10</th>
                <th className="text-right px-3 py-2.5 font-medium text-muted-foreground">NDCG@20</th>
              </tr>
            </thead>
            <tbody>
              {data.regression.map((row) => (
                <tr key={row.model} className={`border-b border-border/30 ${row.model === 'Meta Stack' ? 'bg-emerald-500/5' : ''}`}>
                  <td className={`px-3 py-2.5 font-medium text-sm ${row.model === 'Meta Stack' ? 'text-emerald-400' : ''}`}>
                    {row.model}
                  </td>
                  <MetricCell value={row.rmse} best={bestRMSE} />
                  <MetricCell value={row.mae} best={bestMAE} />
                  <MetricCell value={row.spearman} best={bestSpearman} lowerBetter={false} />
                  <MetricCell value={row.ndcg10} best={bestNDCG10} lowerBetter={false} />
                  <MetricCell value={row.ndcg20} best={bestNDCG20} lowerBetter={false} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Meta Weights */}
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="font-semibold text-sm mb-3">Meta-Ensemble Weights (Ridge Blender)</h3>
        <div className="flex flex-wrap gap-3">
          {data.metaWeights.map((mw) => (
            <div key={mw.model} className="flex items-center gap-2 bg-secondary/50 rounded-lg px-3 py-2">
              <span className="text-xs text-muted-foreground">{mw.model}</span>
              <span className={`text-sm font-mono font-bold ${mw.weight > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {mw.weight > 0 ? '+' : ''}{(mw.weight * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Classification */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="px-5 py-4 border-b border-border bg-secondary/20">
          <h3 className="font-semibold text-sm">Classification Models — Binary Outcome Prediction</h3>
          <div className="flex gap-2 mt-3">
            {data.classification.map((t, i) => (
              <button
                key={t.name}
                onClick={() => setActiveClsTab(i)}
                className={`text-xs px-3 py-1.5 rounded-md font-medium transition-colors ${
                  i === activeClsTab
                    ? 'bg-accent/20 text-accent'
                    : 'bg-secondary/50 text-muted-foreground hover:text-foreground'
                }`}
              >
                {t.name}
              </button>
            ))}
          </div>
        </div>
        <div className="overflow-x-auto">
          <div className="px-5 py-2 text-xs text-muted-foreground">
            Base rate: {(activeTarget.baseRate * 100).toFixed(1)}% — lower Brier/ECE = better, higher AUC = better
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left px-3 py-2.5 font-medium text-muted-foreground">Model</th>
                <th className="text-right px-3 py-2.5 font-medium text-muted-foreground">Brier Score</th>
                <th className="text-right px-3 py-2.5 font-medium text-muted-foreground">Log Loss</th>
                <th className="text-right px-3 py-2.5 font-medium text-muted-foreground">AUC</th>
                <th className="text-right px-3 py-2.5 font-medium text-muted-foreground">ECE</th>
              </tr>
            </thead>
            <tbody>
              {activeTarget.rows.map((row) => (
                <tr key={row.model} className={`border-b border-border/30 ${row.model === 'Meta Stack' ? 'bg-emerald-500/5' : ''}`}>
                  <td className={`px-3 py-2.5 font-medium text-sm ${row.model === 'Meta Stack' ? 'text-emerald-400' : ''}`}>
                    {row.model}
                  </td>
                  <MetricCell value={row.brier} best={bestBrier} />
                  <MetricCell value={row.logLoss} best={bestLL} />
                  <MetricCell value={row.auc} best={bestAUC} lowerBetter={false} />
                  <MetricCell value={row.ece} best={bestECE} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="font-semibold text-sm mb-1">Top Feature Drivers (LightGBM Gain)</h3>
        <p className="text-xs text-muted-foreground mb-4">What the models rely on most when predicting Strokes Gained</p>
        <div className="space-y-2">
          {data.topFeatures.map((f, i) => {
            const maxImp = data.topFeatures[0].importance;
            const pct = (f.importance / maxImp) * 100;
            return (
              <div key={f.name} className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground w-5 text-right font-mono">{i + 1}</span>
                <span className="text-xs w-44 shrink-0 truncate">{f.name}</span>
                <div className="flex-1 h-3 bg-secondary/50 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-accent/60"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
