"use client";

import { AlertTriangle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { Performance } from "@/lib/data/types";
import { formatMaybePctMetric, formatNumber, formatPct } from "@/lib/format";

type ThresholdRow = {
  sport: string;
  segment: string;
  edgeThreshold: number | null;
  minConfidence: number | null;
  sample: number | null;
  accuracy: number | null;
  roi: number | null;
  units: number | null;
};

function numberValue(row: Record<string, string | number | null>, keys: string[]) {
  for (const key of keys) {
    const value = row[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function thresholdRows(records: Performance[]): ThresholdRow[] {
  return records.flatMap((record) =>
    (record.thresholdPerformance ?? []).map((row) => ({
      sport: record.sport,
      segment: String(row.edge_bucket ?? row.segment ?? row.strategy_id ?? "threshold"),
      edgeThreshold: numberValue(row, ["edge_threshold"]),
      minConfidence: numberValue(row, ["min_confidence"]),
      sample: numberValue(row, ["n_bets", "bets", "games", "sample_size"]),
      accuracy: numberValue(row, ["accuracy"]),
      roi: numberValue(row, ["roi"]),
      units: numberValue(row, ["units"]),
    })),
  );
}

export function PerformanceTables({ records }: { records: Performance[] }) {
  const thresholds = thresholdRows(records);
  const gateBadge = (status: "pass" | "warning" | "blocked") => {
    if (status === "pass") return "accent";
    if (status === "blocked") return "destructive";
    return "outline";
  };
  const statusBadge = (status: Performance["productionStatus"]) => {
    if (status === "approved") return "accent";
    if (status === "blocked") return "destructive";
    return "outline";
  };

  return (
    <Tabs defaultValue="roi" className="w-full">
      <TabsList>
        <TabsTrigger value="roi">ROI</TabsTrigger>
        <TabsTrigger value="metrics">Metrics</TabsTrigger>
        <TabsTrigger value="gates">Gates</TabsTrigger>
        <TabsTrigger value="thresholds">Thresholds</TabsTrigger>
        <TabsTrigger value="warnings">Warnings</TabsTrigger>
      </TabsList>

      <TabsContent value="roi">
        <Table className="table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead>Sport</TableHead>
              <TableHead>Market</TableHead>
              <TableHead>Season</TableHead>
              <TableHead>Bets</TableHead>
              <TableHead>W-L-P</TableHead>
              <TableHead>ROI</TableHead>
              <TableHead>Units</TableHead>
              <TableHead>Odds</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {records.map((record) => (
              <TableRow key={`${record.sport}-${record.market}`}>
                <TableCell className="font-medium">{record.sport}</TableCell>
                <TableCell>{record.market}</TableCell>
                <TableCell>{record.season}</TableCell>
                <TableCell>{formatNumber(record.bets)}</TableCell>
                <TableCell>
                  {record.wins == null && record.losses == null && record.pushes == null
                    ? "n/a"
                    : `${formatNumber(record.wins)}-${formatNumber(record.losses)}-${formatNumber(record.pushes)}`}
                </TableCell>
                <TableCell>{formatPct(record.roi)}</TableCell>
                <TableCell>{formatNumber(record.units, 2)}</TableCell>
                <TableCell><Badge variant="outline">{record.oddsStatus}</Badge></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TabsContent>

      <TabsContent value="metrics">
        <Table className="table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead>Sport</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Sample</TableHead>
              <TableHead>Accuracy</TableHead>
              <TableHead>AUC</TableHead>
              <TableHead>Brier</TableHead>
              <TableHead>Log Loss</TableHead>
              <TableHead>MAE</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {records.map((record) => (
              <TableRow key={`${record.sport}-${record.modelVersion}`}>
                <TableCell className="font-medium">{record.sport}</TableCell>
                <TableCell>{record.modelVersion}</TableCell>
                <TableCell>{formatNumber(record.sampleSize)}</TableCell>
                <TableCell>{formatMaybePctMetric(record.metrics.accuracy)}</TableCell>
                <TableCell>{formatNumber(record.metrics.auc as number | null, 4)}</TableCell>
                <TableCell>{formatNumber(record.metrics.brier as number | null, 4)}</TableCell>
                <TableCell>{formatNumber(record.metrics.logLoss as number | null, 4)}</TableCell>
                <TableCell>{formatNumber(record.metrics.mae as number | null, 3)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TabsContent>

      <TabsContent value="gates">
        <Table className="table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead>Sport</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Gate</TableHead>
              <TableHead>Result</TableHead>
              <TableHead className="w-[34%]">Detail</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {records.flatMap((record) =>
              record.productionGates.map((gate) => (
                <TableRow key={`${record.sport}-${gate.id}`}>
                  <TableCell className="font-medium">{record.sport}</TableCell>
                  <TableCell>{record.modelVersion}</TableCell>
                  <TableCell>
                    <Badge variant={statusBadge(record.productionStatus)}>
                      {record.productionStatus}
                    </Badge>
                  </TableCell>
                  <TableCell>{gate.label}</TableCell>
                  <TableCell>
                    <Badge variant={gateBadge(gate.status)}>{gate.status}</Badge>
                  </TableCell>
                  <TableCell className="whitespace-normal text-sm text-muted-foreground">
                    {gate.detail}
                  </TableCell>
                </TableRow>
              )),
            )}
          </TableBody>
        </Table>
      </TabsContent>

      <TabsContent value="thresholds">
        {thresholds.length ? (
          <Table className="table-fixed">
            <TableHeader>
              <TableRow>
                <TableHead>Sport</TableHead>
                <TableHead>Segment</TableHead>
                <TableHead>Edge</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Sample</TableHead>
                <TableHead>Accuracy</TableHead>
                <TableHead>ROI</TableHead>
                <TableHead>Units</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {thresholds.map((row, index) => (
                <TableRow key={`${row.sport}-${row.segment}-${index}`}>
                  <TableCell className="font-medium">{row.sport}</TableCell>
                  <TableCell>{row.segment}</TableCell>
                  <TableCell>{formatNumber(row.edgeThreshold, 1)}</TableCell>
                  <TableCell>{formatMaybePctMetric(row.minConfidence)}</TableCell>
                  <TableCell>{formatNumber(row.sample)}</TableCell>
                  <TableCell>{formatMaybePctMetric(row.accuracy)}</TableCell>
                  <TableCell>{formatPct(row.roi)}</TableCell>
                  <TableCell>{formatNumber(row.units, 2)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <div className="rounded-lg border border-dashed border-border p-5 text-sm text-muted-foreground">
            No threshold or mode performance arrays are available in the current local artifact.
          </div>
        )}
      </TabsContent>

      <TabsContent value="warnings">
        <div className="space-y-3">
          {records.map((record) => (
            <div key={record.sport} className="rounded-lg border border-border p-4">
              <div className="flex items-center gap-2 font-medium">
                <AlertTriangle className="size-4 text-accent" />
                {record.sport}
                <Badge variant="outline">{record.oddsStatus}</Badge>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {record.gaps.length ? (
                  record.gaps.map((gap) => (
                    <Badge key={gap} variant="missing">{gap}</Badge>
                  ))
                ) : (
                  <Badge variant="accent">No blocking gaps recorded</Badge>
                )}
              </div>
            </div>
          ))}
        </div>
      </TabsContent>
    </Tabs>
  );
}
