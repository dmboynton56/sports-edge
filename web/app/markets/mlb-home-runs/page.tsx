import { Activity, AlertTriangle, DollarSign, LineChart, Percent } from "lucide-react";

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
import { getMlbHomeRunFeed } from "@/lib/data/player-markets";
import { formatDateTime, formatNumber, formatPct } from "@/lib/format";

function formatAmerican(price: number | null | undefined) {
  if (typeof price !== "number" || !Number.isFinite(price)) return "n/a";
  return price > 0 ? `+${price}` : `${price}`;
}

export default async function MlbHomeRunsPage() {
  const feed = await getMlbHomeRunFeed();
  const rows = feed.predictions.toSorted(
    (a, b) => ((b.modelProbability ?? 0) - (a.modelProbability ?? 0)),
  );
  const best = rows[0];
  const flagged = rows.filter((row) => (row.qualityFlags?.length ?? 0) > 0).length;
  const rowsWithOdds = rows.filter((row) => row.oddsStatus && row.oddsStatus !== "missing_odds");
  const positiveEdges = rows.filter((row) => (row.edge ?? 0) > 0).length;
  const bestEdge = rows
    .filter((row) => typeof row.edge === "number")
    .toSorted((a, b) => ((b.edge ?? -Infinity) - (a.edge ?? -Infinity)))[0];

  return (
    <div>
      <PageHeader
        title="MLB Home Runs"
        description="Daily probability-first batter HR candidates from projected lineups, probable pitchers, venue context, and recent hitter form."
        meta={feed.generatedAt}
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        <MetricCard
          title="Candidates"
          value={formatNumber(rows.length)}
          detail={`${feed.productionStatus} model surface`}
          icon={LineChart}
          tone={rows.length ? "accent" : "warning"}
        />
        <MetricCard
          title="Top Probability"
          value={best ? formatPct(best.modelProbability) : "n/a"}
          detail={best ? `${best.player} vs ${best.opponent}` : "No slate rows"}
          icon={Activity}
        />
        <MetricCard
          title="Rows With Flags"
          value={formatNumber(flagged)}
          detail="Flags cover lineup, pitcher, and player-history gaps."
          icon={AlertTriangle}
          tone={flagged ? "warning" : "accent"}
        />
        <MetricCard
          title="Odds Coverage"
          value={formatPct(rows.length ? rowsWithOdds.length / rows.length : null)}
          detail={`${formatNumber(rowsWithOdds.length)} candidates with sportsbook odds`}
          icon={Percent}
          tone={rowsWithOdds.length ? "accent" : "warning"}
        />
        <MetricCard
          title="Positive Edges"
          value={formatNumber(positiveEdges)}
          detail={bestEdge ? `${bestEdge.player} ${formatAmerican(bestEdge.bestPrice)}` : "No priced edges"}
          icon={DollarSign}
          tone={positiveEdges ? "accent" : "default"}
        />
      </div>

      {feed.gaps.length ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {feed.gaps.map((gap) => (
            <Badge key={gap} variant="missing">
              {gap}
            </Badge>
          ))}
        </div>
      ) : null}

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Home Run Probability Board</CardTitle>
        </CardHeader>
        <CardContent>
          <Table className="table-fixed">
            <TableHeader>
              <TableRow>
                <TableHead className="w-16">Rank</TableHead>
                <TableHead>Player</TableHead>
                <TableHead>Game</TableHead>
                <TableHead>Slot</TableHead>
                <TableHead>Pitcher</TableHead>
                <TableHead>HR Prob</TableHead>
                <TableHead>Best Price</TableHead>
                <TableHead>Market Prob</TableHead>
                <TableHead>Edge / EV</TableHead>
                <TableHead>Odds</TableHead>
                <TableHead>Baseline</TableHead>
                <TableHead>Flags</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow key={row.id}>
                  <TableCell>{row.rank}</TableCell>
                  <TableCell>
                    <div className="font-medium">{row.player}</div>
                    <div className="text-xs text-muted-foreground">{row.team}</div>
                  </TableCell>
                  <TableCell>
                    <div>{row.opponent}</div>
                    <div className="text-xs text-muted-foreground">{formatDateTime(row.eventTime)}</div>
                  </TableCell>
                  <TableCell>
                    <div>{row.lineupSlot ?? "n/a"}</div>
                    <div className="text-xs text-muted-foreground">{row.lineupStatus}</div>
                  </TableCell>
                  <TableCell className="truncate">{row.opposingProbablePitcher ?? "n/a"}</TableCell>
                  <TableCell className="font-mono font-semibold">{formatPct(row.modelProbability)}</TableCell>
                  <TableCell>
                    <div className="font-mono font-semibold">{formatAmerican(row.bestPrice)}</div>
                    <div className="text-xs text-muted-foreground">{row.bestBook ?? "missing"}</div>
                  </TableCell>
                  <TableCell>
                    <div className="font-mono">{formatPct(row.marketProbability ?? row.impliedProbability)}</div>
                    <div className="text-xs text-muted-foreground">
                      {row.noVigProbability != null ? "no-vig" : row.impliedProbability != null ? "raw" : "n/a"}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div
                      className={
                        row.edge != null && row.edge > 0
                          ? "font-mono font-semibold text-emerald-400"
                          : row.edge != null && row.edge < 0
                            ? "font-mono text-red-400"
                            : "font-mono text-muted-foreground"
                      }
                    >
                      {row.edge != null && row.edge > 0 ? "+" : ""}
                      {formatPct(row.edge)}
                    </div>
                    <div className="text-xs text-muted-foreground">EV {formatPct(row.ev)}</div>
                  </TableCell>
                  <TableCell>
                    <Badge variant={row.oddsStatus === "missing_odds" ? "missing" : "outline"}>
                      {row.oddsStatus ?? "model only"}
                    </Badge>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {formatNumber(row.oddsBooksCount)} books | {formatDateTime(row.oddsSnapshotTs)}
                    </div>
                  </TableCell>
                  <TableCell className="font-mono text-muted-foreground">
                    {formatPct(row.baselineProbability)}
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {(row.qualityFlags ?? []).length ? (
                        row.qualityFlags?.map((flag) => (
                          <Badge key={flag} variant="outline">
                            {flag}
                          </Badge>
                        ))
                      ) : (
                        <Badge variant="accent">clean</Badge>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
