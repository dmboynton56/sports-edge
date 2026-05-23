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

export function PerformanceTables({ records }: { records: Performance[] }) {
  return (
    <Tabs defaultValue="roi" className="w-full">
      <TabsList>
        <TabsTrigger value="roi">ROI</TabsTrigger>
        <TabsTrigger value="metrics">Metrics</TabsTrigger>
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

      <TabsContent value="thresholds">
        <div className="rounded-lg border border-dashed border-border p-5 text-sm text-muted-foreground">
          {records.some((record) => record.thresholdPerformance?.length || record.modePerformance?.length)
            ? "Threshold/mode artifacts are present but not yet shaped for display."
            : "No threshold or mode performance arrays are available in the current local artifact."}
        </div>
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
