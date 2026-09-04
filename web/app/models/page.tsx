import { ChannelCard } from "@/components/dashboard/ChannelCard";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { deriveDataQuality } from "@/lib/data/data-quality";
import { getEvaluationsBundle } from "@/lib/data/evaluations";
import { getPerformanceHistory } from "@/lib/data/performance";
import { getMlbHomeRunBoardSnapshot } from "@/lib/data/player-markets";
import { getResultsData } from "@/lib/data/results";
import { formatNumber } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function ModelsOverviewPage() {
  const [registry, performance, results, mlbHealth] = await Promise.all([
    getEvaluationsBundle(),
    getPerformanceHistory(),
    getResultsData(),
    getMlbHomeRunBoardSnapshot(),
  ]);
  const quality = deriveDataQuality(performance);
  const productionModels = registry.registry.filter((entry) => entry.status === "production").length;
  const performanceRows = performance.records.reduce((total, row) => total + (row.sampleSize ?? 0), 0);
  const resultRows = results.gameRows.length + results.mlbHrRows.length + results.pgaRows.length;
  const qualityIssues = quality.filter((row) => row.status !== "ok").length
    + (mlbHealth.status === "healthy" ? 0 : 1);

  return (
    <div>
      <PageHeader
        title="Models"
        description="Accountability for what is running, how it performed, what was graded, and whether its inputs are healthy."
        meta={`${formatNumber(registry.registry.length)} registered models`}
      />

      <div className="grid gap-3 md:grid-cols-2">
        <ChannelCard
          href="/models/registry"
          title="Registry"
          description="Active model versions, evaluation runs, and strategy evidence."
          figures={[
            { value: formatNumber(registry.registry.length), label: "Registered" },
            { value: formatNumber(productionModels), label: "Production" },
          ]}
          cta="Inspect the registry"
        />
        <ChannelCard
          href="/models/performance"
          title="Performance"
          description="Backtests, live windows, ROI evidence, and production gates by sport."
          figures={[
            { value: formatNumber(performance.records.length), label: "Records" },
            { value: formatNumber(performanceRows), label: "Backtest rows" },
          ]}
          cta="Review performance"
        />
        <ChannelCard
          href="/models/results"
          title="Results"
          description="Official outcomes graded against immutable pregame snapshots."
          figures={[
            { value: formatNumber(results.summaries.length), label: "Summaries" },
            { value: formatNumber(resultRows), label: "Recent grades" },
          ]}
          cta="Open graded results"
        />
        <ChannelCard
          href="/models/data-quality"
          title="Data quality"
          description="Freshness, coverage, environment readiness, and blocking feed gaps."
          figures={[
            { value: formatNumber(qualityIssues), label: "Need attention", tone: qualityIssues ? "down" : "up" },
            { value: formatNumber(quality.length + 1), label: "Sources" },
          ]}
          cta="Check data health"
        />
      </div>
    </div>
  );
}
