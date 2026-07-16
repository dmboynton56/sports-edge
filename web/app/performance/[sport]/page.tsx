import Link from "next/link";
import { notFound } from "next/navigation";

import { PageHeader } from "@/components/dashboard/PageHeader";
import {
  type PerformanceTrendPoint,
} from "@/components/dashboard/PerformanceTrendChartClient";
import { PerformanceTrendChart } from "@/components/dashboard/PerformanceTrendChart";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  getEvaluationHistory,
  getEvaluationRuns,
  type EvaluationRow,
} from "@/lib/data/evaluations";
import {
  bucketWeeklyResults,
  filterByWindow,
  getGameResultRows,
  getMlbHomeRunResultRows,
  getPgaResultRows,
  summarizeGameResults,
  summarizeMlbHr,
  type GameResultRow,
  type PgaResultRow,
  type ResultsSummary,
  type ResultsWindow,
} from "@/lib/data/results";
import { formatDate, formatMaybePctMetric, formatNumber, formatPct } from "@/lib/format";
import { getSport, type MarketEntry, type SportEntry } from "@/lib/markets-registry";
import { cn } from "@/lib/utils";

export const dynamic = "force-dynamic";

const WINDOWS: ResultsWindow[] = ["7d", "30d", "season", "all"];

function isResultsWindow(value: string | string[] | undefined): value is ResultsWindow {
  return typeof value === "string" && WINDOWS.includes(value as ResultsWindow);
}

function MarketHeader({ market }: { market: MarketEntry }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <CardTitle>{market.label}</CardTitle>
      <Link href={market.href} className={cn(buttonVariants({ variant: "outline", size: "sm" }))}>
        Open market
      </Link>
    </div>
  );
}

function GapBadges({ gaps }: { gaps: string[] }) {
  if (!gaps.length) return null;
  return (
    <div className="mb-4 flex flex-wrap gap-2">
      {[...new Set(gaps)].map((gap) => <Badge key={gap} variant="missing">{gap}</Badge>)}
    </div>
  );
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-border bg-muted/20 px-4 py-10 text-center text-sm text-muted-foreground">
      {children}
    </div>
  );
}

function ResultsSummaryTable({ rows }: { rows: ResultsSummary[] }) {
  if (!rows.length) return <EmptyState>No graded results are available in this window.</EmptyState>;
  return (
    <Table className="table-fixed">
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead>Market</TableHead>
          <TableHead>Sample</TableHead>
          <TableHead>W-L-P</TableHead>
          <TableHead>Hit rate</TableHead>
          <TableHead>Flat -110 ROI</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row) => (
          <TableRow key={`${row.market}-${row.modelVersion}`}>
            <TableCell className="font-medium">{row.modelVersion}</TableCell>
            <TableCell>{row.market}</TableCell>
            <TableCell>{formatNumber(row.sample)}</TableCell>
            <TableCell>{row.wins}-{row.losses}-{row.pushes}</TableCell>
            <TableCell>{formatPct(row.hitRate)}</TableCell>
            <TableCell>{formatPct(row.roi)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function weeklyPoints(rows: GameResultRow[]): PerformanceTrendPoint[] {
  const groups = new Map<string, GameResultRow[]>();
  for (const row of rows) {
    groups.set(row.model_version, [...(groups.get(row.model_version) ?? []), row]);
  }
  return Array.from(groups.entries()).flatMap(([model, modelRows]) =>
    bucketWeeklyResults(modelRows).map((bucket) => ({
      label: `${bucket.weekStart.slice(5)} · ${model}`,
      hitRate: bucket.hitRate,
      units: bucket.units,
    })),
  );
}

function TeamMarketCard({ sport, rows }: { sport: SportEntry; rows: GameResultRow[] }) {
  const market = sport.markets[0];
  const summaries = summarizeGameResults(rows);
  return (
    <Card>
      <CardHeader><MarketHeader market={market} /></CardHeader>
      <CardContent className="space-y-6">
        <ResultsSummaryTable rows={summaries} />
        <div>
          <h3 className="mb-3 text-sm font-medium">Weekly ATS hit rate &amp; flat units</h3>
          <PerformanceTrendChart data={weeklyPoints(rows)} mode="weekly" />
        </div>
      </CardContent>
    </Card>
  );
}

function MlbCards({ sport, gameRows, hrRows }: {
  sport: SportEntry;
  gameRows: GameResultRow[];
  hrRows: Awaited<ReturnType<typeof getMlbHomeRunResultRows>>["rows"];
}) {
  const winner = sport.markets.find((market) => market.slug === "winners")!;
  const homeRuns = sport.markets.find((market) => market.slug === "home-runs")!;
  const winnerRows = summarizeGameResults(gameRows).filter((row) => row.market === "winner");
  const hrSummaries = summarizeMlbHr(hrRows);
  return (
    <div className="grid gap-4">
      <Card>
        <CardHeader><MarketHeader market={winner} /></CardHeader>
        <CardContent><ResultsSummaryTable rows={winnerRows} /></CardContent>
      </Card>
      <Card>
        <CardHeader><MarketHeader market={homeRuns} /></CardHeader>
        <CardContent>
          <p className="mb-4 text-sm text-muted-foreground">Hit rates are split by top-k bucket and model version.</p>
          <ResultsSummaryTable rows={hrSummaries} />
        </CardContent>
      </Card>
    </div>
  );
}

type PgaEventSummary = {
  event: string;
  model: string;
  sample: number;
  top10Rate: number | null;
  top20Rate: number | null;
  winnerHits: number;
  expectedWins: number;
};

function booleanRate(rows: PgaResultRow[], field: "top10_hit" | "top20_hit") {
  const graded = rows.filter((row) => typeof row[field] === "boolean");
  return graded.length ? graded.filter((row) => row[field] === true).length / graded.length : null;
}

function pgaEventSummaries(rows: PgaResultRow[]): PgaEventSummary[] {
  const groups = new Map<string, PgaResultRow[]>();
  for (const row of rows) {
    const key = `${row.event_key}|${row.model_version}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return Array.from(groups.entries()).map(([key, group]) => {
    const [event, model] = key.split("|");
    return {
      event,
      model,
      sample: group.length,
      top10Rate: booleanRate(group, "top10_hit"),
      top20Rate: booleanRate(group, "top20_hit"),
      winnerHits: group.filter((row) => row.winner_hit === true).length,
      expectedWins: group.reduce((sum, row) => sum + (row.win_prob ?? 0), 0),
    };
  });
}

function PgaCard({ sport, rows }: { sport: SportEntry; rows: PgaResultRow[] }) {
  const summaries = pgaEventSummaries(rows);
  return (
    <Card>
      <CardHeader><MarketHeader market={sport.markets[0]} /></CardHeader>
      <CardContent>
        {summaries.length ? (
          <Table className="table-fixed">
            <TableHeader><TableRow>
              <TableHead>Event</TableHead><TableHead>Model</TableHead><TableHead>Players</TableHead>
              <TableHead>Top 10</TableHead><TableHead>Top 20</TableHead><TableHead>Winner actual / expected</TableHead>
            </TableRow></TableHeader>
            <TableBody>{summaries.map((row) => (
              <TableRow key={`${row.event}-${row.model}`}>
                <TableCell className="font-medium">{row.event}</TableCell>
                <TableCell>{row.model}</TableCell>
                <TableCell>{formatNumber(row.sample)}</TableCell>
                <TableCell>{formatPct(row.top10Rate)}</TableCell>
                <TableCell>{formatPct(row.top20Rate)}</TableCell>
                <TableCell>{row.winnerHits} / {formatNumber(row.expectedWins, 2)}</TableCell>
              </TableRow>
            ))}</TableBody>
          </Table>
        ) : <EmptyState>No graded PGA events are available in this window.</EmptyState>}
      </CardContent>
    </Card>
  );
}

function EmptySportCards({ sport }: { sport: SportEntry }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {sport.markets.map((market) => (
        <Card key={market.slug} className="border-dashed bg-muted/20">
          <CardHeader><MarketHeader market={market} /></CardHeader>
          <CardContent><EmptyState>No graded markets yet.</EmptyState></CardContent>
        </Card>
      ))}
    </div>
  );
}

function statusVariant(status: string) {
  if (status === "approved") return "accent" as const;
  if (status === "rejected") return "destructive" as const;
  return "outline" as const;
}

function evaluationTrend(rows: EvaluationRow[]) {
  const metric = rows.some((row) => row.displayMetrics.roi != null) ? "roi" : "accuracy";
  const points = rows
    .filter((row) => row.displayMetrics[metric] != null)
    .toSorted((a, b) => a.generated_at.localeCompare(b.generated_at))
    .map((row) => ({
      label: `${row.generated_at.slice(0, 10)} · ${row.model_version}`,
      metric: row.displayMetrics[metric],
    }));
  return { metric, points };
}

function Backtests({ runs, history, gaps }: { runs: EvaluationRow[]; history: EvaluationRow[]; gaps: string[] }) {
  const trend = evaluationTrend(history);
  return (
    <section className="mt-8">
      <div className="mb-4">
        <h2 className="text-xl font-semibold">Persisted backtests</h2>
        <p className="mt-1 text-sm text-muted-foreground">Evaluation runs stored in Supabase, separate from live grading.</p>
      </div>
      <GapBadges gaps={gaps} />
      <Card>
        <CardHeader><CardTitle>Evaluation runs</CardTitle></CardHeader>
        <CardContent>
          {runs.length ? (
            <Table className="table-fixed">
              <TableHeader><TableRow>
                <TableHead>Evaluation</TableHead><TableHead>Model</TableHead><TableHead>Test range</TableHead>
                <TableHead>Accuracy</TableHead><TableHead>AUC</TableHead><TableHead>Brier / log loss</TableHead><TableHead>ROI</TableHead><TableHead>Status</TableHead>
              </TableRow></TableHeader>
              <TableBody>{runs.map((row, index) => (
                <TableRow key={`${row.evaluation_name}-${row.model_version}-${row.generated_at}-${index}`}>
                  <TableCell className="font-medium">{row.evaluation_name}</TableCell>
                  <TableCell>{row.model_version}</TableCell>
                  <TableCell>{formatDate(row.test_start_date)}–{formatDate(row.test_end_date)}</TableCell>
                  <TableCell>{formatMaybePctMetric(row.displayMetrics.accuracy)}</TableCell>
                  <TableCell>{formatNumber(row.displayMetrics.auc, 4)}</TableCell>
                  <TableCell>{formatNumber(row.displayMetrics.brier, 4)} / {formatNumber(row.displayMetrics.logLoss, 4)}</TableCell>
                  <TableCell>{formatPct(row.displayMetrics.roi)}</TableCell>
                  <TableCell><Badge variant={statusVariant(row.status)}>{row.status}</Badge></TableCell>
                </TableRow>
              ))}</TableBody>
            </Table>
          ) : <EmptyState>No persisted evaluation runs are available for this sport.</EmptyState>}
        </CardContent>
      </Card>
      {history.length ? (
        <Card className="mt-4">
          <CardHeader><CardTitle>{trend.metric === "roi" ? "ROI" : "Accuracy"} over time</CardTitle></CardHeader>
          <CardContent><PerformanceTrendChart data={trend.points} mode="metric" metricLabel={trend.metric === "roi" ? "ROI" : "Accuracy"} /></CardContent>
        </Card>
      ) : null}
    </section>
  );
}

export default async function SportPerformancePage({
  params,
  searchParams,
}: {
  params: Promise<{ sport: string }>;
  searchParams: Promise<{ window?: string | string[] }>;
}) {
  const [{ sport: sportSlug }, query] = await Promise.all([params, searchParams]);
  const sport = getSport(sportSlug);
  if (!sport) notFound();

  const window = isResultsWindow(query.window) ? query.window : "season";
  const league = sport.slug.toUpperCase();
  const [evaluations, history] = await Promise.all([
    getEvaluationRuns(league),
    getEvaluationHistory(league),
  ]);

  let marketContent: React.ReactNode;
  let resultGaps: string[] = [];
  if (sport.slug === "nba" || sport.slug === "nfl") {
    const result = await getGameResultRows(league);
    const rows = filterByWindow(result.rows, window, "game_date");
    resultGaps = result.gaps;
    marketContent = <TeamMarketCard sport={sport} rows={rows} />;
  } else if (sport.slug === "mlb") {
    const [games, homeRuns] = await Promise.all([getGameResultRows("MLB"), getMlbHomeRunResultRows()]);
    resultGaps = [...games.gaps, ...homeRuns.gaps];
    marketContent = (
      <MlbCards
        sport={sport}
        gameRows={filterByWindow(games.rows, window, "game_date")}
        hrRows={filterByWindow(homeRuns.rows, window, "game_date")}
      />
    );
  } else if (sport.slug === "pga") {
    const result = await getPgaResultRows();
    resultGaps = result.gaps;
    marketContent = <PgaCard sport={sport} rows={filterByWindow(result.rows, window, "evaluated_at")} />;
  } else {
    marketContent = <EmptySportCards sport={sport} />;
  }

  const seasonLabel = String(new Date().getUTCFullYear());
  const windowLabels: Record<ResultsWindow, string> = { "7d": "7 days", "30d": "30 days", season: seasonLabel, all: "All" };

  return (
    <div>
      <PageHeader title={`${sport.label} Performance`} description="Windowed live grades by market, plus persisted model evaluation evidence." />
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <span className="mr-1 text-sm text-muted-foreground">Results window</span>
        {WINDOWS.map((item) => (
          <Link
            key={item}
            href={`/performance/${sport.slug}?window=${item}`}
            aria-current={window === item ? "page" : undefined}
            className={cn(buttonVariants({ variant: window === item ? "default" : "outline", size: "sm" }))}
          >
            {windowLabels[item]}
          </Link>
        ))}
        <Badge variant="outline" className="ml-auto">Season = current calendar year</Badge>
      </div>
      <GapBadges gaps={resultGaps} />
      {marketContent}
      <Backtests runs={evaluations.rows} history={history.rows} gaps={[...evaluations.gaps, ...history.gaps]} />
    </div>
  );
}
