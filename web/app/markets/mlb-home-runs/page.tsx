import { Activity, AlertTriangle, LineChart } from "lucide-react";

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

export default async function MlbHomeRunsPage() {
  const feed = await getMlbHomeRunFeed();
  const rows = feed.predictions.toSorted(
    (a, b) => ((b.modelProbability ?? 0) - (a.modelProbability ?? 0)),
  );
  const best = rows[0];
  const flagged = rows.filter((row) => (row.qualityFlags?.length ?? 0) > 0).length;

  return (
    <div>
      <PageHeader
        title="MLB Home Runs"
        description="Daily probability-first batter HR candidates from projected lineups, probable pitchers, venue context, and recent hitter form."
        meta={feed.generatedAt}
      />

      <div className="grid gap-4 sm:grid-cols-3">
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
