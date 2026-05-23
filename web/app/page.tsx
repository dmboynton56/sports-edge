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
import { getLocalPredictions } from "@/lib/data/predictions";
import { formatNumber, formatPct, formatPctFromWhole } from "@/lib/format";

export default async function Home() {
  const [history, predictionFeed] = await Promise.all([
    getPerformanceHistory(),
    getLocalPredictions(),
  ]);
  const quality = deriveDataQuality(history);
  const activeWarnings = quality.filter((row) => row.status !== "ok").length;
  const roiRecords = history.records.filter((record) => typeof record.roi === "number");
  const bestRoi = roiRecords.toSorted((a, b) => (b.roi ?? -Infinity) - (a.roi ?? -Infinity))[0];

  return (
    <div>
      <PageHeader
        title="Operations Overview"
        description="A dense control surface for pre-live markets, model performance, ROI history, and data source readiness."
        meta={history.generatedAt}
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Active Market Edges"
          value={formatNumber(predictionFeed.predictions.length)}
          detail={predictionFeed.gaps[0] ?? "Loaded from local prediction artifact."}
          icon={LineChart}
          tone={predictionFeed.predictions.length ? "accent" : "warning"}
        />
        <MetricCard
          title="Tracked Models"
          value={formatNumber(history.records.length)}
          detail="Cross-sport performance records in the current artifact."
          icon={Activity}
        />
        <MetricCard
          title="Best Available ROI"
          value={bestRoi ? formatPct(bestRoi.roi) : "n/a"}
          detail={bestRoi ? `${bestRoi.sport} ${bestRoi.market}` : "ROI is missing from all records."}
          icon={LineChart}
          tone={bestRoi?.roi && bestRoi.roi > 0 ? "accent" : "default"}
        />
        <MetricCard
          title="Data Warnings"
          value={formatNumber(activeWarnings)}
          detail="Warnings include partial odds coverage and missing source history."
          icon={DatabaseZap}
          tone={activeWarnings ? "warning" : "accent"}
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
