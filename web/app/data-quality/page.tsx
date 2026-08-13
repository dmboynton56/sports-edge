import { AlertTriangle, CheckCircle2, DatabaseZap, XCircle } from "lucide-react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { PageHeader, SectionHeading } from "@/components/dashboard/PageHeader";
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
import { getMlbHomeRunBoardSnapshot } from "@/lib/data/player-markets";
import { getSupabaseMissingEnv } from "@/lib/data/supabase";
import { formatDateTime, formatNumber, formatPctFromWhole } from "@/lib/format";
import { sportColor } from "@/lib/sports";
import { cn } from "@/lib/utils";

export const dynamic = "force-dynamic";

const STATUS_BADGE = {
  ok: { variant: "positive" as const, label: "Healthy" },
  warning: { variant: "warning" as const, label: "Partial" },
  missing: { variant: "destructive" as const, label: "Missing" },
  blocked: { variant: "destructive" as const, label: "Blocked" },
};

/** Gaps arrive as "NBA: …" strings; grouping them makes the list scannable. */
function groupGaps(gaps: string[]) {
  const groups = new Map<string, string[]>();
  for (const gap of gaps) {
    const [head, ...rest] = gap.split(": ");
    const hasPrefix = rest.length > 0 && head.length <= 5;
    const key = hasPrefix ? head : "General";
    const body = hasPrefix ? rest.join(": ") : gap;
    groups.set(key, [...(groups.get(key) ?? []), body]);
  }
  return [...groups.entries()];
}

export default async function DataQualityPage() {
  const [history, mlbHr] = await Promise.all([
    getPerformanceHistory(),
    getMlbHomeRunBoardSnapshot(),
  ]);
  const rows = deriveDataQuality(history);
  const missingSupabase = getSupabaseMissingEnv();
  const missingBigQuery = getBigQueryMissingEnv();
  const blocked = rows.filter(
    (row) => row.status === "blocked" || row.status === "missing",
  ).length;
  const warnings = rows.filter((row) => row.status === "warning").length;
  const envGaps = missingSupabase.length + missingBigQuery.length;
  const gapGroups = groupGaps(history.gaps);

  return (
    <div>
      <PageHeader
        title="Data quality"
        description="What each upstream source is actually delivering, and what the models are missing because of it."
        meta={history.generatedAt}
      />

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Sources tracked"
          value={formatNumber(rows.length)}
          detail="Derived from the performance artifact."
          icon={DatabaseZap}
        />
        <MetricCard
          title="Partial coverage"
          value={formatNumber(warnings)}
          detail="Reporting, but not on every game."
          icon={AlertTriangle}
          tone={warnings ? "warning" : "accent"}
        />
        <MetricCard
          title="Missing entirely"
          value={formatNumber(blocked)}
          detail="No odds or no source history at all."
          icon={XCircle}
          tone={blocked ? "warning" : "accent"}
        />
        <MetricCard
          title="Unset env vars"
          value={formatNumber(envGaps)}
          detail={envGaps ? "Live feeds fall back to local artifacts." : "All runtime variables are set."}
          icon={CheckCircle2}
          tone={envGaps ? "warning" : "accent"}
        />
      </div>

      <Card className="mt-3">
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>Trusted MLB HR board</CardTitle>
            <p className="mt-1 text-sm text-muted-foreground">
              Current serving run for {mlbHr.slateDate}; the website fails closed when this contract is stale.
            </p>
          </div>
          <Badge variant={mlbHr.status === "healthy" ? "positive" : mlbHr.status === "partial" ? "warning" : "missing"}>
            {mlbHr.status}
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 text-sm sm:grid-cols-4">
            <div><div className="text-xs text-muted-foreground">Candidates</div><div className="font-semibold">{formatNumber(mlbHr.counts.candidates)}</div></div>
            <div><div className="text-xs text-muted-foreground">Priced</div><div className="font-semibold">{formatNumber(mlbHr.counts.priced)}</div></div>
            <div><div className="text-xs text-muted-foreground">Top-25 coverage</div><div className="font-semibold">{formatPctFromWhole(mlbHr.counts.top25Coverage == null ? null : mlbHr.counts.top25Coverage * 100)}</div></div>
            <div><div className="text-xs text-muted-foreground">Last completed</div><div className="font-semibold">{formatDateTime(mlbHr.completedAt)}</div></div>
          </div>
          {mlbHr.gaps.length ? <div className="mt-4 flex flex-wrap gap-2">{mlbHr.gaps.map((gap) => <Badge key={gap} variant="missing">{gap}</Badge>)}</div> : null}
        </CardContent>
      </Card>

      <SectionHeading title="Coverage by source" note="Share of graded games that have odds attached" />

      <Card className="overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Source</TableHead>
              <TableHead className="text-right">Coverage</TableHead>
              <TableHead className="hidden text-right sm:table-cell">Missing rows</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="hidden md:table-cell">Updated</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => {
              const status = STATUS_BADGE[row.status] ?? STATUS_BADGE.warning;
              return (
                <TableRow key={row.source}>
                  <TableCell>
                    <div className="flex items-center gap-2.5">
                      {row.sport ? (
                        <span
                          className={cn(
                            "h-4 w-[3px] shrink-0 rounded-full",
                            sportColor(row.sport).fill,
                          )}
                        />
                      ) : null}
                      <span className="font-semibold text-foreground">{row.source}</span>
                    </div>
                    {row.notes ? (
                      <div className="mt-0.5 pl-[22px] font-mono text-xs text-muted-foreground">
                        {row.notes}
                      </div>
                    ) : null}
                  </TableCell>
                  <TableCell className="text-right font-semibold text-foreground">
                    {formatPctFromWhole(row.coveragePct)}
                  </TableCell>
                  <TableCell className="hidden text-right sm:table-cell">
                    {formatNumber(row.missingRows)}
                  </TableCell>
                  <TableCell>
                    <Badge variant={status.variant}>{status.label}</Badge>
                  </TableCell>
                  <TableCell className="hidden whitespace-nowrap text-xs md:table-cell">
                    {formatDateTime(row.lastUpdated)}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Card>

      <div className="mt-3 grid gap-3 lg:grid-cols-[1.4fr_0.6fr]">
        <Card>
          <CardHeader>
            <CardTitle>What&apos;s blocking each league</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            {gapGroups.map(([league, items]) => (
              <div key={league}>
                <div className="flex items-center gap-2.5">
                  <span
                    className={cn(
                      "h-4 w-[3px] rounded-full",
                      sportColor(league).fill,
                    )}
                  />
                  <span className="font-display text-base font-bold tracking-tight">
                    {league}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {items.length} {items.length === 1 ? "issue" : "issues"}
                  </span>
                </div>
                <ul className="mt-2 space-y-1.5 pl-[22px]">
                  {items.map((item) => (
                    <li
                      key={item}
                      className="text-[13px] leading-relaxed text-muted-foreground"
                    >
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Environment</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <div>
              <div className="mb-2 text-sm font-semibold">Supabase</div>
              <div className="flex flex-wrap gap-1.5">
                {missingSupabase.length ? (
                  missingSupabase.map((env) => (
                    <Badge key={env} variant="missing" className="font-mono text-[11px]">
                      {env}
                    </Badge>
                  ))
                ) : (
                  <Badge variant="positive">Configured</Badge>
                )}
              </div>
            </div>
            <div>
              <div className="mb-2 text-sm font-semibold">BigQuery</div>
              <div className="flex flex-wrap gap-1.5">
                {missingBigQuery.length ? (
                  missingBigQuery.map((env) => (
                    <Badge key={env} variant="missing" className="font-mono text-[11px]">
                      {env}
                    </Badge>
                  ))
                ) : (
                  <Badge variant="positive">Configured</Badge>
                )}
              </div>
            </div>
            {envGaps ? (
              <p className="text-[13px] leading-relaxed text-muted-foreground">
                Without these, the dashboard reads the exported artifacts in{" "}
                <span className="font-mono text-xs">public/data</span> instead of the
                live warehouse.
              </p>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
