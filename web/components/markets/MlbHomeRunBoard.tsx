"use client";

import { useState } from "react";
import { Activity, AlertTriangle, Clock3 } from "lucide-react";

import { EmptyState } from "@/components/dashboard/EmptyState";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { Notice } from "@/components/dashboard/Notice";
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
import {
  getMlbHomeRunModelLabel,
  type MlbHomeRunPrediction,
  type MlbHrBoardSnapshot,
} from "@/lib/data/mlb-hr-board";
import { formatDateTime, formatNumber, formatPct } from "@/lib/format";
import { formatFairAmericanOdds } from "@/lib/fair-odds";

const PAGE_SIZE = 25;

function StatusBadge({ status }: { status: MlbHrBoardSnapshot["status"] }) {
  const config = {
    healthy: ["positive", "Healthy"],
    partial: ["warning", "Partial"],
    stale: ["missing", "Stale"],
    unavailable: ["destructive", "Unavailable"],
    no_slate: ["outline", "No slate"],
  } as const;
  const [variant, label] = config[status];
  return <Badge variant={variant}>{label}</Badge>;
}

function BoardMetrics({ snapshot }: { snapshot: MlbHrBoardSnapshot }) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <MetricCard
        title="Candidates"
        value={formatNumber(snapshot.counts.candidates)}
        detail="all eligible model rows"
        icon={Activity}
        tone={snapshot.counts.candidates ? "accent" : "warning"}
      />
      <MetricCard
        title="Model status"
        value="Candidate"
        detail={getMlbHomeRunModelLabel("mlb-hr-v1")}
        icon={Activity}
      />
      <MetricCard
        title="Run window"
        value={snapshot.runWindow}
        detail={snapshot.completedAt ? `completed ${formatDateTime(snapshot.completedAt)}` : "not completed"}
        icon={Clock3}
      />
    </div>
  );
}

function FairOddsCell({ row }: { row: MlbHomeRunPrediction }) {
  return (
    <div>
      <div className="font-mono font-semibold">{formatFairAmericanOdds(row.modelProbability)}</div>
      <div className="text-xs text-muted-foreground">Model fair odds</div>
    </div>
  );
}

function CandidateCard({ row }: { row: MlbHomeRunPrediction }) {
  return (
    <article className="rounded-xl border border-border bg-card p-4 shadow-soft">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
            Rank #{row.rank ?? "—"}
          </div>
          <h3 className="mt-1 font-display text-lg font-bold tracking-tight">{row.player}</h3>
          <p className="text-sm text-muted-foreground">
            {row.team ?? "—"} vs {row.opponent ?? "—"}
          </p>
        </div>
        <Badge variant="outline">Model only</Badge>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div>
          <div className="text-xs text-muted-foreground">HR probability</div>
          <div className="font-mono font-semibold">{formatPct(row.modelProbability)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Fair odds</div>
          <FairOddsCell row={row} />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span>{row.eventTime ? formatDateTime(row.eventTime) : "Event time pending"}</span>
        {row.updatedAt ? <span>Model {formatDateTime(row.updatedAt)}</span> : null}
        {(row.qualityFlags ?? []).map((flag) => (
          <Badge key={flag} variant="outline">{flag}</Badge>
        ))}
      </div>
    </article>
  );
}

function DesktopRows({ rows }: { rows: MlbHomeRunPrediction[] }) {
  return (
    <div className="hidden overflow-x-auto md:block">
      <Table className="min-w-[700px]">
        <TableHeader>
          <TableRow>
            <TableHead className="w-16">Rank</TableHead>
            <TableHead>Player / game</TableHead>
            <TableHead>HR probability</TableHead>
            <TableHead>Fair odds</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Model as of</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.id}>
              <TableCell className="font-mono">{row.rank ?? "—"}</TableCell>
              <TableCell>
                <div className="font-semibold">{row.player}</div>
                <div className="text-xs text-muted-foreground">
                  {row.team ?? "—"} vs {row.opponent ?? "—"} · {row.eventTime ? formatDateTime(row.eventTime) : "time pending"}
                </div>
              </TableCell>
              <TableCell className="font-mono font-semibold">{formatPct(row.modelProbability)}</TableCell>
              <TableCell><FairOddsCell row={row} /></TableCell>
              <TableCell><Badge variant="outline">Model only</Badge></TableCell>
              <TableCell className="text-xs text-muted-foreground">{row.updatedAt ? formatDateTime(row.updatedAt) : "—"}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function MlbHomeRunBoard({ snapshot }: { snapshot: MlbHrBoardSnapshot }) {
  const [page, setPage] = useState(0);
  const pageCount = Math.max(1, Math.ceil(snapshot.rows.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount - 1);
  const pageRows = snapshot.rows.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE);

  if (snapshot.status === "stale" || snapshot.status === "unavailable" || snapshot.status === "no_slate") {
    const title = snapshot.status === "no_slate"
      ? "No MLB games on this slate"
      : snapshot.status === "stale"
        ? "Board updating"
        : "Board unavailable";
    const description = snapshot.status === "no_slate"
      ? "The schedule was checked and no games were confirmed for today."
      : "Candidate rows stay hidden until a current, validated Supabase run is available.";
    return (
      <div className="mt-4 space-y-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between gap-3">
              <div><StatusBadge status={snapshot.status} /></div>
              <span className="text-xs text-muted-foreground">Slate {snapshot.slateDate}</span>
            </div>
            <EmptyState className="border-0 bg-transparent px-0 pb-0 pt-8" title={title} description={description} />
          </CardContent>
        </Card>
        <Notice title="Why rows are hidden" items={snapshot.gaps.length ? snapshot.gaps : ["No current board health record was returned."]} />
      </div>
    );
  }

  return (
    <div className="mt-4 space-y-4">
      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle className="flex flex-wrap items-center gap-2">
              MLB HR probabilities <StatusBadge status={snapshot.status} />
            </CardTitle>
            <p className="mt-2 text-sm text-muted-foreground">
              Slate {snapshot.slateDate} · Candidate model · {snapshot.runWindow} refresh
            </p>
          </div>
          <div className="text-left text-xs text-muted-foreground sm:text-right">
            <div>Predictions {formatDateTime(snapshot.predictionAsOf)}</div>
          </div>
        </CardHeader>
        <CardContent>
          <BoardMetrics snapshot={snapshot} />
        </CardContent>
      </Card>

      <Notice title="About fair odds" items={["Fair odds are calculated from the model's home run probability. They are not sportsbook offers, and no betting edge or EV is calculated for these rows."]} />
      {snapshot.gaps.length ? <Notice title="Data-source health" items={snapshot.gaps} /> : null}

      <Card>
        <CardHeader className="gap-3">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle>All current candidates</CardTitle>
              <p className="mt-1 text-sm text-muted-foreground">Rows for games within five minutes of first pitch are hidden.</p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0 sm:px-5 sm:pb-5">
          {pageRows.length ? (
            <>
              <DesktopRows rows={pageRows} />
              <div className="space-y-3 px-5 md:hidden">
                {pageRows.map((row) => <CandidateCard key={row.id} row={row} />)}
              </div>
              <div className="flex items-center justify-between border-t border-border px-5 py-4 text-sm text-muted-foreground">
                <span>{snapshot.rows.length} {snapshot.rows.length === 1 ? "candidate" : "candidates"}</span>
                <div className="flex items-center gap-2">
                  <button type="button" className="rounded-md border border-border px-2.5 py-1 disabled:opacity-40" disabled={currentPage === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</button>
                  <span>Page {currentPage + 1} of {pageCount}</span>
                  <button type="button" className="rounded-md border border-border px-2.5 py-1 disabled:opacity-40" disabled={currentPage >= pageCount - 1} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>Next</button>
                </div>
              </div>
            </>
          ) : (
            <EmptyState className="border-0 bg-transparent py-12" title="No current candidates" description="Check again after the next model refresh." />
          )}
        </CardContent>
      </Card>

      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <AlertTriangle className="size-3.5" />
        <span>Model fair odds are derived from the published probability and do not include a sportsbook margin.</span>
      </div>
    </div>
  );
}
