import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { MlbVerticalSummary } from "@/lib/data/mlb-vertical";

const LABELS: Record<string, string> = {
  moneyline: "Moneyline",
  run_line: "Run line",
  total: "Totals",
  pitcher_strikeouts: "Pitcher strikeouts",
  batter_home_runs: "Batter home runs",
};

function number(value: unknown, digits = 3) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function percent(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

function modelMetric(market: string, metrics: Record<string, unknown> | undefined) {
  if (!metrics) return "—";
  if (market === "total") {
    const binaryHeads = metrics.binary_heads as Record<string, Record<string, unknown>> | undefined;
    const head = binaryHeads?.["8.5"]?.model as Record<string, unknown> | undefined;
    return head?.brier == null ? "—" : `Brier ${number(head.brier)}`;
  }
  if (market === "batter_home_runs") {
    const heldout = metrics.heldout_test as Record<string, unknown> | undefined;
    return heldout?.brier == null ? "—" : `Brier ${number(heldout.brier)}`;
  }
  const brier = metrics.brier;
  if (brier != null) return `Brier ${number(brier)}`;
  if (metrics.mae != null) return `MAE ${number(metrics.mae)}`;
  const regression = metrics.regression as Record<string, unknown> | undefined;
  if (regression?.mae != null) return `MAE ${number(regression.mae)}`;
  const test = metrics.test as Record<string, unknown> | undefined;
  const testModel = test?.model as Record<string, unknown> | undefined;
  return testModel?.mae == null ? "—" : `MAE ${number(testModel.mae)}`;
}

export function MlbEvaluationSummary({ summary }: { summary: MlbVerticalSummary }) {
  const markets = Object.entries(summary.markets ?? {});
  const edges = summary.edges;
  const testGames = summary.markets?.moneyline?.test_rows;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Held-out games</CardTitle></CardHeader>
          <CardContent className="text-2xl font-bold">{testGames?.toLocaleString("en-US") ?? "—"}</CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Edge rows</CardTitle></CardHeader>
          <CardContent className="text-2xl font-bold">{edges?.rows?.toLocaleString("en-US") ?? "—"}</CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Free odds coverage</CardTitle></CardHeader>
          <CardContent className="text-2xl font-bold">{percent(summary.markets?.moneyline?.odds?.coverage)}</CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Positive EV moneylines</CardTitle></CardHeader>
          <CardContent className="text-2xl font-bold">{edges?.positive_ev_moneylines?.toLocaleString("en-US") ?? "—"}</CardContent>
        </Card>
      </div>

      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle>Model evaluation and market availability</CardTitle>
          <p className="text-sm text-muted-foreground">
            Time-split test metrics; statistical signals without a price are not sportsbook EV.
          </p>
        </CardHeader>
        <CardContent className="overflow-x-auto p-0">
          <table className="w-full min-w-[680px] text-left text-sm">
            <thead className="border-y border-border bg-muted/40 text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-5 py-3">Market</th>
                <th className="px-5 py-3">Test rows</th>
                <th className="px-5 py-3">Metric</th>
                <th className="px-5 py-3">Odds</th>
                <th className="px-5 py-3">Gate</th>
              </tr>
            </thead>
            <tbody>
              {markets.map(([market, value]) => {
                const gate = value.quality_gate?.status ?? "review";
                const gateVariant = gate === "candidate" ? "positive" : gate === "blocked" ? "destructive" : "warning";
                return (
                  <tr key={market} className="border-b border-border/70 last:border-0">
                    <td className="px-5 py-3 font-semibold">{LABELS[market] ?? market}</td>
                    <td className="px-5 py-3 tabular-nums">{value.test_rows?.toLocaleString("en-US") ?? "—"}</td>
                    <td className="px-5 py-3">{modelMetric(market, value.metrics)}</td>
                    <td className="px-5 py-3 text-muted-foreground">
                      {market === "moneyline" ? `${percent(value.odds?.coverage)} joined` : "Not available"}
                    </td>
                    <td className="px-5 py-3"><Badge variant={gateVariant}>{gate}</Badge></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}
