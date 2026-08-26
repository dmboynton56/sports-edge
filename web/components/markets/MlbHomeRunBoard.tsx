"use client";

import { useMemo, useState } from "react";
import { Activity, AlertTriangle, Clock3, DollarSign, ShieldCheck } from "lucide-react";

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
import { isFiniteNumber } from "@/lib/data/json";
import { formatDateTime, formatNumber, formatPct } from "@/lib/format";

type Filter = "all" | "priced" | "model-only";
const PAGE_SIZE = 25;
const FILTERS = ["all", "priced", "model-only"] satisfies Filter[];

function formatAmerican(price: number | null | undefined) {
  if (!isFiniteNumber(price) || price === 0) return "n/a";
  return price > 0 ? `+${price}` : `${price}`;
}

function isPriced(row: MlbHomeRunPrediction) {
  return row.oddsStatus === "ok" || row.oddsStatus === "raw_implied";
}

function StatusBadge({ status }: { status: MlbHrBoardSnapshot["status"] }) {
  const config = {
    healthy: ["positive", "Healthy"],
    partial: ["warning", "Partial coverage"],
    stale: ["missing", "Stale"],
    unavailable: ["destructive", "Unavailable"],
    no_slate: ["outline", "No slate"],
  } as const;
  const [variant, label] = config[status];
  return <Badge variant={variant}>{label}</Badge>;
}


function PriceCell({ row }: { row: MlbHomeRunPrediction }) {
  if (!isPriced(row)) {
    return <span className="tag">model only</span>;
  }
  return (
    <div>
      <div className="num">{formatAmerican(row.bestPrice ?? row.price)}</div>
      <div className="tag">{row.bestBook ?? row.book}</div>
    </div>
  );
}

function EdgeCell({ row }: { row: MlbHomeRunPrediction }) {
  if (!isPriced(row)) return <span className="text-muted-foreground">—</span>;
  return (
    <div>
      <div className={row.edge != null && row.edge > 0 ? "num text-positive" : "num"}>
        {formatPct(row.edge)}
      </div>
      <div className="tag">ev {formatPct(row.ev)}</div>
    </div>
  );
}

function CandidateCard({ row }: { row: MlbHomeRunPrediction }) {
  const priced = isPriced(row);
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
        <Badge variant={priced ? "positive" : "missing"}>{priced ? "Priced" : "Model only"}</Badge>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div>
          <div className="text-xs text-muted-foreground">HR probability</div>
          <div className="font-mono font-semibold">{formatPct(row.modelProbability)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Price</div>
          <PriceCell row={row} />
        </div>
        {priced ? (
          <>
            <div>
              <div className="text-xs text-muted-foreground">Market probability</div>
              <div className="font-mono">{formatPct(row.marketProbability ?? row.impliedProbability)}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Edge / EV</div>
              <EdgeCell row={row} />
            </div>
          </>
        ) : null}
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span>{row.eventTime ? formatDateTime(row.eventTime) : "Event time pending"}</span>
        {row.oddsSnapshotTs ? <span>Odds {formatDateTime(row.oddsSnapshotTs)}</span> : null}
        {(row.qualityFlags ?? []).map((flag) => (
          <Badge key={flag} variant="outline">{flag}</Badge>
        ))}
      </div>
    </article>
  );
}

function DesktopRows({ rows }: { rows: MlbHomeRunPrediction[] }) {
  return (
    <div className="overflow-x-auto">
      <Table className="min-w-[900px]">
        <TableHeader>
          <TableRow>
            <TableHead className="w-12">
              <span className="tag">rank</span>
            </TableHead>
            <TableHead>
              <span className="tag">player / game</span>
            </TableHead>
            <TableHead>
              <span className="tag">hr probability</span>
            </TableHead>
            <TableHead>
              <span className="tag">price</span>
            </TableHead>
            <TableHead>
              <span className="tag">market prob</span>
            </TableHead>
            <TableHead>
              <span className="tag">edge</span>
            </TableHead>
            <TableHead>
              <span className="tag">odds as of</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => {
            const priced = isPriced(row);
            return (
              <TableRow key={row.id}>
                <TableCell className="num">{row.rank ?? "—"}</TableCell>
                <TableCell>
                  <div className="font-medium">{row.player}</div>
                  <div className="tag">
                    {row.team ?? "—"} vs {row.opponent ?? "—"}
                  </div>
                </TableCell>
                <TableCell className="num">{formatPct(row.modelProbability)}</TableCell>
                <TableCell><PriceCell row={row} /></TableCell>
                <TableCell className="num">{priced ? formatPct(row.marketProbability ?? row.impliedProbability) : "—"}</TableCell>
                <TableCell><EdgeCell row={row} /></TableCell>
                <TableCell className="tag">{row.oddsSnapshotTs ? formatDateTime(row.oddsSnapshotTs) : "—"}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

export function MlbHomeRunBoard({ snapshot }: { snapshot: MlbHrBoardSnapshot }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [page, setPage] = useState(0);
  const filtered = useMemo(() => {
    if (filter === "priced") return snapshot.rows.filter(isPriced);
    if (filter === "model-only") return snapshot.rows.filter((row) => !isPriced(row));
    return snapshot.rows;
  }, [filter, snapshot.rows]);
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount - 1);
  const pageRows = filtered.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE);

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
      <div className="space-y-4">
        <div className="border border-border bg-card p-4">
          <div className="flex items-center justify-between gap-3">
            <div><StatusBadge status={snapshot.status} /></div>
            <span className="tag">slate {snapshot.slateDate}</span>
          </div>
          <EmptyState className="border-0 bg-transparent px-0 pb-0 pt-8" title={title} description={description} />
        </div>
        {snapshot.gaps.length ? (
          <Notice title="Why rows are hidden" items={snapshot.gaps.length ? snapshot.gaps : ["No current board health record was returned."]} />
        ) : null}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {snapshot.status === "partial" || (snapshot.counts.candidates > 0 && snapshot.counts.priced === 0) ? (
        <Notice
          tone="warning"
          title={snapshot.counts.priced === 0 ? "No sportsbook prices available" : "Partial pricing coverage"}
          items={
            snapshot.counts.priced === 0
              ? [
                  `All ${snapshot.counts.candidates} candidates are model-only. The serving pipeline completed successfully, but no valid sportsbook prices were available at publication time.`,
                  "This usually means the upstream odds provider (OddsPapi) had no fresh MLB HR markets when the board was generated. Model probabilities remain valid for research.",
                  "Check the data-quality page for OddsPapi validation status and upstream coverage details.",
                ]
              : [
                  `Only ${snapshot.counts.top25Priced} of ${snapshot.counts.top25Eligible || "the eligible top-25"} candidates have fresh prices (${formatPct(snapshot.counts.top25Coverage)} coverage).`,
                  "Model-only rows remain visible for research, but they have no actionable price, edge, EV, or Kelly values.",
                ]
          }
        />
      ) : null}
      {snapshot.gaps.length ? <Notice title="Data-source health" items={snapshot.gaps} /> : null}

      <div className="border border-border bg-card">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="tag">filter</div>
          <div className="flex items-center gap-2">
            {FILTERS.map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => { setFilter(value); setPage(0); }}
                className={`px-2 py-1 text-xs transition-colors ${filter === value ? "font-medium text-foreground" : "text-muted-foreground hover:text-foreground"}`}
              >
                {value === "all" ? "All" : value === "priced" ? "Priced" : "Model-only"}
              </button>
            ))}
          </div>
        </div>
        {pageRows.length ? (
          <>
            <DesktopRows rows={pageRows} />
            <div className="flex items-center justify-between border-t border-border px-4 py-3 text-xs">
              <span className="num">{filtered.length} {filtered.length === 1 ? "candidate" : "candidates"}</span>
              <div className="flex items-center gap-3">
                <button type="button" className="tag disabled:opacity-40" disabled={currentPage === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>previous</button>
                <span className="num">page {currentPage + 1} of {pageCount}</span>
                <button type="button" className="tag disabled:opacity-40" disabled={currentPage >= pageCount - 1} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>next</button>
              </div>
            </div>
          </>
        ) : (
          <EmptyState className="border-0 bg-transparent py-12" title="No rows match this filter" description="Switch to All candidates to inspect the full current model surface." />
        )}
      </div>

      <div className="tag flex items-center gap-2">
        <AlertTriangle className="size-3" />
        <span>Priced rows are snapshots from the publication run. Historical edges are never recalculated from newer odds.</span>
      </div>
    </div>
  );
}
