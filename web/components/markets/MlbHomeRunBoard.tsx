"use client";

import { useMemo, useState } from "react";
import { Activity, AlertTriangle, DollarSign, LineChart, Percent } from "lucide-react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  getMlbHomeRunModelLabel,
  type MlbHomeRunBoardData,
  type MlbHomeRunPrediction,
} from "@/lib/data/mlb-hr-board";
import { formatDateTime, formatGamesSinceLastHr, formatNumber, formatPct } from "@/lib/format";

function formatAmerican(price: number | null | undefined) {
  if (typeof price !== "number" || !Number.isFinite(price)) return "n/a";
  return price > 0 ? `+${price}` : `${price}`;
}

function sortRows(predictions: MlbHomeRunPrediction[]) {
  return predictions.toSorted(
    (a, b) => (b.modelProbability ?? 0) - (a.modelProbability ?? 0),
  );
}

function BoardMetrics({ rows }: { rows: MlbHomeRunPrediction[] }) {
  const best = rows[0];
  const flagged = rows.filter((row) => (row.qualityFlags?.length ?? 0) > 0).length;
  const rowsWithOdds = rows.filter((row) => row.oddsStatus && row.oddsStatus !== "missing_odds");
  const positiveEdges = rows.filter((row) => (row.edge ?? 0) > 0).length;
  const bestEdge = rows
    .filter((row) => typeof row.edge === "number")
    .toSorted((a, b) => (b.edge ?? -Infinity) - (a.edge ?? -Infinity))[0];

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
      <MetricCard
        title="Candidates"
        value={formatNumber(rows.length)}
        detail="candidate model surface"
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
  );
}

function BoardTable({ rows }: { rows: MlbHomeRunPrediction[] }) {
  return (
    <TooltipProvider>
      <Table className="table-fixed">
      <TableHeader>
        <TableRow>
          <TableHead className="w-16">Rank</TableHead>
          <TableHead>Player</TableHead>
          <TableHead className="w-20">Since HR</TableHead>
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
        {rows.map((row, index) => (
          <TableRow key={row.id}>
            <TableCell>{index + 1}</TableCell>
            <TableCell>
              <div className="font-medium">{row.player}</div>
              <div className="text-xs text-muted-foreground">{row.team}</div>
            </TableCell>
            <TableCell>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="cursor-help font-mono text-sm">
                    {formatGamesSinceLastHr(row.gamesSinceLastHr, row.qualityFlags)}
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  {row.lastHrDate
                    ? `Last HR: ${row.lastHrDate}. Based on last 45 days of boxscores.`
                    : "No homer in the loaded boxscore history window (45 days)."}
                </TooltipContent>
              </Tooltip>
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
    </TooltipProvider>
  );
}

type MlbHomeRunBoardProps = {
  board: MlbHomeRunBoardData;
};

export function MlbHomeRunBoard({ board }: MlbHomeRunBoardProps) {
  const initialModel = board.availableModels.includes(board.defaultModel)
    ? board.defaultModel
    : board.availableModels[0] ?? board.defaultModel;
  const [activeModel, setActiveModel] = useState(initialModel);

  const activeFeed = board.models[activeModel];
  const rows = useMemo(
    () => sortRows(activeFeed?.predictions ?? []),
    [activeFeed?.predictions],
  );

  if (!board.availableModels.length) {
    return (
      <div className="mt-4 flex flex-wrap gap-2">
        {(board.models[board.defaultModel]?.gaps ?? ["No MLB home run predictions available."]).map((gap) => (
          <Badge key={gap} variant="missing">
            {gap}
          </Badge>
        ))}
      </div>
    );
  }

  return (
    <Tabs value={activeModel} onValueChange={setActiveModel} className="mt-4">
      <TabsList>
        {board.availableModels.map((modelKey) => (
          <TabsTrigger key={modelKey} value={modelKey}>
            {getMlbHomeRunModelLabel(modelKey)}
          </TabsTrigger>
        ))}
      </TabsList>

      {board.availableModels.map((modelKey) => {
        const modelFeed = board.models[modelKey];
        const modelRows = sortRows(modelFeed?.predictions ?? []);
        return (
          <TabsContent key={modelKey} value={modelKey} className="space-y-4">
            {modelFeed?.gaps.length ? (
              <div className="flex flex-wrap gap-2">
                {modelFeed.gaps.map((gap) => (
                  <Badge key={gap} variant="missing">
                    {gap}
                  </Badge>
                ))}
              </div>
            ) : null}

            <BoardMetrics rows={modelRows} />

            <Card>
              <CardHeader>
                <CardTitle>
                  Home Run Probability Board · {getMlbHomeRunModelLabel(modelKey)}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <BoardTable rows={modelRows} />
              </CardContent>
            </Card>
          </TabsContent>
        );
      })}

      {activeFeed && !rows.length ? (
        <div className="mt-4 text-sm text-muted-foreground">
          No rows available for {getMlbHomeRunModelLabel(activeModel)}.
        </div>
      ) : null}
    </Tabs>
  );
}
