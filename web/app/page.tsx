import { Activity, AlertTriangle, DatabaseZap, LineChart } from "lucide-react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { RoiChart } from "@/components/dashboard/RoiChart";
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
import { deriveDataQuality } from "@/lib/data/data-quality";
import { getPerformanceHistory } from "@/lib/data/performance";
import { getMlbHomeRunBoardData, getProductionPredictionFeed } from "@/lib/data/player-markets";
import { getResultsData } from "@/lib/data/results";
import { formatNumber, formatPct, formatPctFromWhole } from "@/lib/format";

export default async function Home() {
  const [history, predictionFeed, results, hrBoard] = await Promise.all([
    getPerformanceHistory(),
    getProductionPredictionFeed(),
    getResultsData(),
    getMlbHomeRunBoardData(),
  ]);
  const quality = deriveDataQuality(history);
  const gradedSample = results.summaries.reduce((sum, row) => sum + row.sample, 0);
  const bestHitRate = results.summaries
    .filter((row) => typeof row.hitRate === "number")
    .toSorted((a, b) => (b.hitRate ?? -Infinity) - (a.hitRate ?? -Infinity))[0];

  return (
    <div>
      <PageHeader
        title="Operations Overview"
        description="A dense control surface for pre-live markets, model performance, ROI history, and data source readiness."
        meta={history.generatedAt}
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Today's Slate"
          value={formatNumber(predictionFeed.predictions.length)}
          detail={predictionFeed.gaps[0] ?? "Predictions loaded across live market surfaces."}
          icon={LineChart}
          tone={predictionFeed.predictions.length ? "accent" : "warning"}
        />
        <MetricCard
          title="Graded Results"
          value={formatNumber(gradedSample)}
          detail="Rows across ATS, winner, HR, and PGA result tables."
          icon={Activity}
          tone={gradedSample ? "accent" : "warning"}
        />
        <MetricCard
          title="Best Hit Rate"
          value={bestHitRate ? formatPct(bestHitRate.hitRate) : "n/a"}
          detail={bestHitRate ? `${bestHitRate.league} ${bestHitRate.market}` : "No graded hit rates yet."}
          icon={LineChart}
          tone={bestHitRate ? "accent" : "default"}
        />
        <MetricCard
          title="Statcast Coverage"
          value={formatPct(hrBoard.statcastHealth?.coverage ?? null)}
          detail={
            hrBoard.statcastHealth
              ? `${formatNumber(hrBoard.statcastHealth.readyRows)} of ${formatNumber(hrBoard.statcastHealth.totalRows)} HR rows ready`
              : "No HR health metadata available."
          }
          icon={DatabaseZap}
          tone={hrBoard.statcastHealth?.artifactLoaded === false ? "warning" : "accent"}
        />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(420px,0.8fr)]">
        <Card>
          <CardHeader>
            <CardTitle>ROI Snapshot</CardTitle>
          </CardHeader>
          <CardContent>
            <RoiChart records={history.records} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Source Readiness</CardTitle>
          </CardHeader>
          <CardContent>
            <Table className="table-fixed">
              <TableHeader>
                <TableRow>
                  <TableHead>Source</TableHead>
                  <TableHead>Coverage</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {quality.slice(0, 6).map((row) => (
                  <TableRow key={row.source}>
                    <TableCell>
                      <div className="truncate font-medium">{row.source}</div>
                      <div className="truncate text-xs text-muted-foreground">{row.notes ?? "n/a"}</div>
                    </TableCell>
                    <TableCell>{formatPctFromWhole(row.coveragePct)}</TableCell>
                    <TableCell>
                      <Badge variant={row.status === "ok" ? "accent" : "missing"}>
                        {row.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Results Snapshot</CardTitle>
          </CardHeader>
          <CardContent>
            <Table className="table-fixed">
              <TableHeader>
                <TableRow>
                  <TableHead>League</TableHead>
                  <TableHead>Market</TableHead>
                  <TableHead>Sample</TableHead>
                  <TableHead>Hit Rate</TableHead>
                  <TableHead>ROI</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.summaries.slice(0, 8).map((row) => (
                  <TableRow key={`${row.league}-${row.market}-${row.modelVersion}`}>
                    <TableCell className="font-medium">{row.league}</TableCell>
                    <TableCell>{row.market}</TableCell>
                    <TableCell>{formatNumber(row.sample)}</TableCell>
                    <TableCell>{formatPct(row.hitRate)}</TableCell>
                    <TableCell>{formatPct(row.roi)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <Table className="table-fixed">
              <TableHeader>
                <TableRow>
                  <TableHead>Sport</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Market</TableHead>
                  <TableHead>ROI</TableHead>
                  <TableHead>Sample</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {history.records.map((record) => (
                  <TableRow key={`${record.sport}-${record.modelVersion}`}>
                    <TableCell className="font-medium">{record.sport}</TableCell>
                    <TableCell>{record.modelVersion}</TableCell>
                    <TableCell>{record.market}</TableCell>
                    <TableCell>{formatPct(record.roi)}</TableCell>
                    <TableCell>{formatNumber(record.sampleSize)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Blocking Gaps</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {history.gaps.length ? (
                history.gaps.slice(0, 14).map((gap) => (
                  <Badge key={gap} variant="missing" className="max-w-full">
                    <AlertTriangle className="mr-1 size-3" />
                    {gap}
                  </Badge>
                ))
              ) : (
                <Badge variant="accent">No blocking gaps recorded</Badge>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
