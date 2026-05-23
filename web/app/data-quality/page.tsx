import { AlertTriangle, CheckCircle2, DatabaseZap } from "lucide-react";

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
import { deriveDataQuality } from "@/lib/data/data-quality";
import { getBigQueryMissingEnv } from "@/lib/data/bigquery";
import { getPerformanceHistory } from "@/lib/data/performance";
import { getSupabaseMissingEnv } from "@/lib/data/supabase";
import { formatNumber, formatPctFromWhole } from "@/lib/format";

export default async function DataQualityPage() {
  const history = await getPerformanceHistory();
  const rows = deriveDataQuality(history);
  const missingSupabase = getSupabaseMissingEnv();
  const missingBigQuery = getBigQueryMissingEnv();
  const blocked = rows.filter((row) => row.status === "blocked" || row.status === "missing").length;
  const warnings = rows.filter((row) => row.status === "warning").length;

  return (
    <div>
      <PageHeader
        title="Data Quality"
        description="Source coverage, odds gaps, sync status, and environment readiness for Supabase, BigQuery, and local exported artifacts."
        meta={history.generatedAt}
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard title="Sources" value={formatNumber(rows.length)} detail="Derived from the performance artifact." icon={DatabaseZap} />
        <MetricCard title="Warnings" value={formatNumber(warnings)} detail="Partial coverage or validation issues." icon={AlertTriangle} tone={warnings ? "warning" : "accent"} />
        <MetricCard title="Missing/Blocked" value={formatNumber(blocked)} detail="Missing odds or unavailable source history." icon={AlertTriangle} tone={blocked ? "warning" : "accent"} />
        <MetricCard title="Env Gaps" value={formatNumber(missingSupabase.length + missingBigQuery.length)} detail="Web runtime variables not configured." icon={CheckCircle2} tone={missingSupabase.length + missingBigQuery.length ? "warning" : "accent"} />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <Card>
          <CardHeader>
            <CardTitle>Coverage Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            <Table className="table-fixed">
              <TableHeader>
                <TableRow>
                  <TableHead>Source</TableHead>
                  <TableHead>Coverage</TableHead>
                  <TableHead>Missing Rows</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Updated</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => (
                  <TableRow key={row.source}>
                    <TableCell>
                      <div className="font-medium">{row.source}</div>
                      <div className="text-xs text-muted-foreground">{row.notes ?? "n/a"}</div>
                    </TableCell>
                    <TableCell>{formatPctFromWhole(row.coveragePct)}</TableCell>
                    <TableCell>{formatNumber(row.missingRows)}</TableCell>
                    <TableCell>
                      <Badge variant={row.status === "ok" ? "accent" : "missing"}>
                        {row.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">{row.lastUpdated ?? "n/a"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Environment</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="mb-2 text-sm font-medium">Supabase web env</div>
              <div className="flex flex-wrap gap-2">
                {missingSupabase.length ? (
                  missingSupabase.map((env) => <Badge key={env} variant="missing">{env}</Badge>)
                ) : (
                  <Badge variant="accent">configured</Badge>
                )}
              </div>
            </div>
            <div>
              <div className="mb-2 text-sm font-medium">BigQuery server env</div>
              <div className="flex flex-wrap gap-2">
                {missingBigQuery.length ? (
                  missingBigQuery.map((env) => <Badge key={env} variant="missing">{env}</Badge>)
                ) : (
                  <Badge variant="accent">configured</Badge>
                )}
              </div>
            </div>
            <div>
              <div className="mb-2 text-sm font-medium">Blocking gaps</div>
              <div className="flex flex-wrap gap-2">
                {history.gaps.map((gap) => (
                  <Badge key={gap} variant="missing">{gap}</Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
