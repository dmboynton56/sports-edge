import { Activity, Brain, DatabaseZap, Target, TrendingUp } from "lucide-react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getMlbHrExperimentSummary, type ExperimentMetricSet } from "@/lib/data/mlb-hr-experiment";
import { formatNumber, formatPct } from "@/lib/format";

function formatMetric(value?: number | null, digits = 4) {
  if (typeof value !== "number" || Number.isNaN(value)) return "pending";
  return value.toFixed(digits);
}

function formatDelta(value?: number | null, digits = 4, lowerIsBetter = true) {
  if (typeof value !== "number" || Number.isNaN(value)) return "pending";
  const good = lowerIsBetter ? value < 0 : value > 0;
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)} ${good ? "better" : "worse"}`;
}

function modelLabel(model: ExperimentMetricSet | null) {
  if (!model) return "missing";
  return `${model.modelVersion ?? "unknown"} / ${model.estimator ?? "unknown"}`;
}

export default async function MlbHrPytorchInsightPage() {
  const summary = await getMlbHrExperimentSummary();
  const baseline = summary.baseline;
  const pytorch = summary.pytorch;
  const blend = summary.pytorchBlend;
  const handedBlend = summary.pytorchHandedBlend;
  const statcastBlend = summary.pytorchStatcastBlend;
  const baselineTest = baseline?.test;
  const pytorchTest = pytorch.test;
  const blendTest = blend?.test;
  const handedBlendTest = handedBlend?.test;
  const statcastBlendTest = statcastBlend?.test;
  const dailyModel = summary.dailyOutcomes?.modelProbability;

  const metricRows = [
    {
      metric: "Brier",
      baseline: formatMetric(baselineTest?.brier),
      pytorch: formatMetric(pytorchTest?.brier),
      blend: formatMetric(blendTest?.brier),
      delta: formatDelta(summary.comparison.brierDelta),
      blendDelta: formatDelta(summary.blendComparison?.brierDelta),
    },
    {
      metric: "Log loss",
      baseline: formatMetric(baselineTest?.logLoss),
      pytorch: formatMetric(pytorchTest?.logLoss),
      blend: formatMetric(blendTest?.logLoss),
      delta: formatDelta(summary.comparison.logLossDelta),
      blendDelta: formatDelta(summary.blendComparison?.logLossDelta),
    },
    {
      metric: "AUC",
      baseline: formatMetric(baselineTest?.auc),
      pytorch: formatMetric(pytorchTest?.auc),
      blend: formatMetric(blendTest?.auc),
      delta: formatDelta(summary.comparison.aucDelta, 4, false),
      blendDelta: formatDelta(summary.blendComparison?.aucDelta, 4, false),
    },
    {
      metric: "Top 10 hit rate",
      baseline: formatPct(baselineTest?.top10HitRate ?? null),
      pytorch: formatPct(pytorchTest?.top10HitRate ?? null),
      blend: formatPct(blendTest?.top10HitRate ?? null),
      delta: formatDelta(summary.comparison.top10HitRateDelta, 4, false),
      blendDelta: formatDelta(summary.blendComparison?.top10HitRateDelta, 4, false),
    },
    {
      metric: "Top 25 hit rate",
      baseline: formatPct(baselineTest?.top25HitRate ?? null),
      pytorch: formatPct(pytorchTest?.top25HitRate ?? null),
      blend: formatPct(blendTest?.top25HitRate ?? null),
      delta: "tracked",
      blendDelta: formatDelta(summary.blendComparison?.top25HitRateDelta, 4, false),
    },
  ];
  const enrichmentRows = [
    handedBlendTest
      ? {
          model: handedBlend?.modelVersion ?? "handedness blend",
          source: "Handedness",
          brier: formatMetric(handedBlendTest.brier),
          logLoss: formatMetric(handedBlendTest.logLoss),
          auc: formatMetric(handedBlendTest.auc),
          top10: formatPct(handedBlendTest.top10HitRate ?? null),
          top25: formatPct(handedBlendTest.top25HitRate ?? null),
          top10Delta: formatDelta(summary.handedBlendComparison?.top10HitRateDelta, 4, false),
          top25Delta: formatDelta(summary.handedBlendComparison?.top25HitRateDelta, 4, false),
        }
      : null,
    statcastBlendTest
      ? {
          model: statcastBlend?.modelVersion ?? "statcast blend",
          source: "Statcast",
          brier: formatMetric(statcastBlendTest.brier),
          logLoss: formatMetric(statcastBlendTest.logLoss),
          auc: formatMetric(statcastBlendTest.auc),
          top10: formatPct(statcastBlendTest.top10HitRate ?? null),
          top25: formatPct(statcastBlendTest.top25HitRate ?? null),
          top10Delta: formatDelta(summary.statcastBlendComparison?.top10HitRateDelta, 4, false),
          top25Delta: formatDelta(summary.statcastBlendComparison?.top25HitRateDelta, 4, false),
        }
      : null,
  ].filter((row): row is NonNullable<typeof row> => row !== null);

  return (
    <div>
      <PageHeader
        title="MLB HR PyTorch Experiment"
        description="A live notebook-style post separating calibrated HR probability from ranking signals for shortlist discovery."
        meta={summary.generatedAt}
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Baseline Brier"
          value={formatMetric(baselineTest?.brier)}
          detail={modelLabel(baseline)}
          icon={Target}
          tone="accent"
        />
        <MetricCard
          title="Baseline AUC"
          value={formatMetric(baselineTest?.auc)}
          detail={`${formatNumber(baselineTest?.rows ?? null)} held-out batter games`}
          icon={Activity}
        />
        <MetricCard
          title="Best Candidate"
          value={blendTest ? formatMetric(blendTest.logLoss) : "pending"}
          detail={blend ? `${modelLabel(blend)} log loss` : modelLabel(pytorch)}
          icon={Brain}
          tone={blendTest ? "accent" : "warning"}
        />
        <MetricCard
          title="Data Expansion"
          value={statcastBlendTest ? formatMetric(statcastBlendTest.auc) : formatNumber(summary.dataExpansion.length)}
          detail={statcastBlendTest ? "Statcast blend AUC" : "Statcast, pybaseball, and MLB boxscore lanes"}
          icon={DatabaseZap}
        />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,0.9fr)_minmax(460px,1.1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Experiment Thesis</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm leading-6 text-muted-foreground">
            <p className="text-foreground">{summary.blogDraft.summary}</p>
            <p>
              The current model is useful because it is leakage-aware and calibrated against a fixed
              held-out slice. The raw PyTorch model is comparable, but the stronger result is the
              validation-chosen PyTorch + heuristic blend, which improves probability metrics while
              preserving the same pregame feature rules. Handedness and Statcast enrichments improve
              ranking in different parts of the board, so the next production decision should keep
              probability quality and candidate ranking separate.
            </p>
            <p className="text-foreground">
              Statcast currently improves broad ranking, handedness improves top-10, v1 remains the
              calibrated production baseline.
            </p>
            <div className="flex flex-wrap gap-2">
              <Badge variant={summary.experimentStatus === "pytorch_evaluated" ? "accent" : "missing"}>
                {summary.experimentStatus}
              </Badge>
              <Badge variant="outline">candidate market</Badge>
              <Badge variant="outline">same split comparison</Badge>
              <Badge variant="outline">ranking experiment</Badge>
              {handedBlend ? <Badge variant="outline">handedness enriched</Badge> : null}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Before / After Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <Table className="table-fixed">
              <TableHeader>
                <TableRow>
                  <TableHead>Metric</TableHead>
                  <TableHead>Random Forest</TableHead>
                  <TableHead>PyTorch</TableHead>
                  <TableHead>Blend</TableHead>
                  <TableHead>Blend Delta</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {metricRows.map((row) => (
                  <TableRow key={row.metric}>
                    <TableCell className="font-medium">{row.metric}</TableCell>
                    <TableCell className="font-mono">{row.baseline}</TableCell>
                    <TableCell className="font-mono">{row.pytorch}</TableCell>
                    <TableCell className="font-mono">{row.blend}</TableCell>
                    <TableCell>{row.blendDelta}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <p className="mt-3 text-xs text-muted-foreground">{summary.comparison.interpretation}</p>
            {blend ? (
              <p className="mt-2 text-xs text-muted-foreground">
                Blend weights: {formatPct(blend.pytorchWeight ?? null)} PyTorch /{" "}
                {formatPct(blend.heuristicWeight ?? null)} heuristic.{" "}
                {summary.blendComparison?.interpretation}
              </p>
            ) : null}
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        {dailyModel ? (
          <Card>
            <CardHeader>
              <CardTitle>Daily Outcome Loop</CardTitle>
            </CardHeader>
            <CardContent>
              <Table className="table-fixed">
                <TableHeader>
                  <TableRow>
                    <TableHead>Rows</TableHead>
                    <TableHead>Brier</TableHead>
                    <TableHead>Log loss</TableHead>
                    <TableHead>AUC</TableHead>
                    <TableHead>Top 10</TableHead>
                    <TableHead>Top 25</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell>{formatNumber(Number(dailyModel.rows ?? 0))}</TableCell>
                    <TableCell className="font-mono">{formatMetric(Number(dailyModel.brier))}</TableCell>
                    <TableCell className="font-mono">{formatMetric(Number(dailyModel.log_loss))}</TableCell>
                    <TableCell className="font-mono">{formatMetric(Number(dailyModel.auc))}</TableCell>
                    <TableCell className="font-mono">{formatPct(Number(dailyModel.top_10_hit_rate))}</TableCell>
                    <TableCell className="font-mono">{formatPct(Number(dailyModel.top_25_hit_rate))}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
              <p className="mt-3 text-xs text-muted-foreground">
                Evaluated dates: {(summary.dailyOutcomes?.evaluatedDates ?? []).join(", ") || "pending"}.
                Missing outcome rows: {formatNumber(summary.dailyOutcomes?.missingOutcomeRows ?? null)}.
                This is live-board feedback for the currently served model, not the held-out training split.
              </p>
            </CardContent>
          </Card>
        ) : null}

        {enrichmentRows.length ? (
          <Card>
            <CardHeader>
            <CardTitle>Enrichment Lanes</CardTitle>
            </CardHeader>
            <CardContent>
              <Table className="table-fixed">
                <TableHeader>
                  <TableRow>
                    <TableHead>Source</TableHead>
                    <TableHead>Brier</TableHead>
                    <TableHead>Log loss</TableHead>
                    <TableHead>AUC</TableHead>
                    <TableHead>Top 10</TableHead>
                    <TableHead>Top 25</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {enrichmentRows.map((row) => (
                    <TableRow key={row.model}>
                      <TableCell>
                        <div className="font-medium">{row.source}</div>
                        <div className="truncate text-xs text-muted-foreground">{row.model}</div>
                      </TableCell>
                      <TableCell className="font-mono">{row.brier}</TableCell>
                      <TableCell className="font-mono">{row.logLoss}</TableCell>
                      <TableCell className="font-mono">{row.auc}</TableCell>
                      <TableCell className="font-mono">{row.top10}</TableCell>
                      <TableCell className="font-mono">{row.top25}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <p className="mt-3 text-xs text-muted-foreground">
                {summary.handedBlendComparison?.interpretation} {summary.statcastBlendComparison?.interpretation}
              </p>
            </CardContent>
          </Card>
        ) : null}

        <Card>
          <CardHeader>
            <CardTitle>Data We Want Next</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {summary.dataExpansion.map((source) => (
                <div key={source.name} className="border-b border-border pb-4 last:border-0 last:pb-0">
                  <a className="font-medium text-accent hover:underline" href={source.url}>
                    {source.name}
                  </a>
                  <p className="mt-1 text-sm leading-6 text-muted-foreground">{source.use}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Publish Gate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {summary.blogDraft.publishWhen.map((gate) => (
                <div key={gate} className="flex gap-3 text-sm text-muted-foreground">
                  <TrendingUp className="mt-0.5 size-4 shrink-0 text-accent" />
                  <span>{gate}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
