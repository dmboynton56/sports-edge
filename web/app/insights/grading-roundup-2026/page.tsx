import Link from "next/link";

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
import { getPerformanceHistory } from "@/lib/data/performance";
import { getResultsData } from "@/lib/data/results";
import type { Performance } from "@/lib/data/types";
import { formatNumber, formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

function hasCalibration(record: Performance) {
  return [record.metrics.auc, record.metrics.brier, record.metrics.logLoss].some(
    (value) => typeof value === "number" && Number.isFinite(value),
  );
}

function bestMeasuredRoi(records: Performance[]) {
  const values = records
    .map((record) => record.roi)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return values.length ? Math.max(...values) : null;
}

export default async function GradingRoundupInsightPage() {
  const [data, performance] = await Promise.all([
    getResultsData(),
    getPerformanceHistory(),
  ]);
  const recordsBySport = new Map<string, Performance[]>();

  for (const record of performance.records) {
    recordsBySport.set(record.sport, [...(recordsBySport.get(record.sport) ?? []), record]);
  }

  const sportSnapshots = Array.from(recordsBySport.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([sport, records]) => ({
      sport,
      bestRoi: bestMeasuredRoi(records),
      calibrated: records.filter(hasCalibration).length,
      records: records.length,
      approved: records.filter((record) => record.productionStatus === "approved").length,
      candidate: records.filter((record) => record.productionStatus === "candidate").length,
      blocked: records.filter((record) => record.productionStatus === "blocked").length,
    }));

  return (
    <div>
      <PageHeader
        title="2026 Grading & Backtest Roundup"
        description="Season results computed live from graded result tables, paired with the latest persisted backtest evidence."
        meta={data.generatedAt}
      />

      <Card>
        <CardHeader>
          <CardTitle>What&apos;s now graded automatically</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-6 text-muted-foreground">
          <p>
            NBA, NFL, and MLB spread and winner outcomes are graded in the daily workflows,
            alongside MLB home run outcomes. PGA placement results are graded in the tournament
            workflow.
          </p>
          <p>
            Those grades persist to Supabase, giving the dashboard a durable results history
            instead of a point-in-time pipeline view.
          </p>
        </CardContent>
      </Card>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Graded results to date</CardTitle>
        </CardHeader>
        <CardContent>
          {data.gaps.length ? (
            <div className="mb-4 flex flex-wrap gap-2">
              {data.gaps.map((gap) => (
                <Badge key={gap} variant="missing">
                  {gap}
                </Badge>
              ))}
            </div>
          ) : null}

          {data.summaries.length ? (
            <Table className="table-fixed">
              <TableHeader>
                <TableRow>
                  <TableHead>League</TableHead>
                  <TableHead>Market</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Sample</TableHead>
                  <TableHead>W-L-P</TableHead>
                  <TableHead>Hit Rate</TableHead>
                  <TableHead>Flat ROI</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.summaries.map((row) => (
                  <TableRow key={`${row.league}-${row.market}-${row.modelVersion}`}>
                    <TableCell className="font-medium">{row.league}</TableCell>
                    <TableCell>{row.market}</TableCell>
                    <TableCell>{row.modelVersion}</TableCell>
                    <TableCell>{formatNumber(row.sample)}</TableCell>
                    <TableCell>
                      {row.wins}-{row.losses}-{row.pushes}
                    </TableCell>
                    <TableCell>{formatPct(row.hitRate)}</TableCell>
                    <TableCell>{formatPct(row.roi)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">No graded rows yet.</p>
          )}
        </CardContent>
      </Card>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Backtest snapshot</CardTitle>
        </CardHeader>
        <CardContent>
          {sportSnapshots.length ? (
            <Table className="table-fixed">
              <TableHeader>
                <TableRow>
                  <TableHead>Sport</TableHead>
                  <TableHead>Best measured ROI</TableHead>
                  <TableHead>Calibration available</TableHead>
                  <TableHead>Approved</TableHead>
                  <TableHead>Candidate</TableHead>
                  <TableHead>Blocked</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sportSnapshots.map((snapshot) => (
                  <TableRow key={snapshot.sport}>
                    <TableCell className="font-medium">{snapshot.sport}</TableCell>
                    <TableCell>{formatPct(snapshot.bestRoi)}</TableCell>
                    <TableCell>
                      {formatNumber(snapshot.calibrated)} of {formatNumber(snapshot.records)}
                    </TableCell>
                    <TableCell>{formatNumber(snapshot.approved)}</TableCell>
                    <TableCell>{formatNumber(snapshot.candidate)}</TableCell>
                    <TableCell>{formatNumber(snapshot.blocked)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">No persisted backtest rows yet.</p>
          )}
        </CardContent>
      </Card>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Where to look next</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-x-6 gap-y-3 text-sm">
          <Link className="font-medium text-accent hover:underline" href="/performance">
            Performance hub
          </Link>
          <Link className="font-medium text-accent hover:underline" href="/results">
            Graded results
          </Link>
          <Link className="font-medium text-accent hover:underline" href="/markets">
            Markets
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}
