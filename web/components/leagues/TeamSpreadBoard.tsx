"use client";

import Link from "next/link";

import { PickCard, type EnrichedPick } from "@/components/PickCard";
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
import type { FreshnessStatus, TeamSlateFeed, TeamSlateGame } from "@/lib/data/team-markets";
import { formatDateTime, formatNumber, formatPct } from "@/lib/format";

function formatSpread(line: number | null) {
  if (line == null || !Number.isFinite(line)) return "n/a";
  return line > 0 ? `+${line.toFixed(1)}` : line.toFixed(1);
}

function freshnessVariant(status: FreshnessStatus) {
  if (status === "fresh") return "accent";
  if (status === "no_odds") return "outline";
  return "missing";
}

function toEnrichedPick(game: TeamSlateGame): EnrichedPick | null {
  if (game.modelSpread == null || game.homeWinProb == null || game.bookSpread == null) {
    return null;
  }
  return {
    id: game.gameId,
    game: {
      id: game.gameId,
      league: game.league,
      homeTeam: game.homeTeam,
      awayTeam: game.awayTeam,
      gameTimeUtc: game.gameTimeUtc,
    },
    currentOdds: {
      book: "consensus",
      market: "spread",
      line: game.bookSpread,
      price: -110,
    },
    prediction: {
      predictedSpread: game.modelSpread,
      homeWinProb: game.homeWinProb,
    },
    edgePts: game.edgePts ?? 0,
  };
}

function FreshnessBadge({ status }: { status: FreshnessStatus }) {
  const labels: Record<FreshnessStatus, string> = {
    fresh: "Fresh",
    stale: "Stale",
    no_prediction: "No pred",
    no_odds: "No odds",
  };
  return <Badge variant={freshnessVariant(status)}>{labels[status]}</Badge>;
}

export function TeamSpreadBoard({
  feed,
  detailBasePath,
}: {
  feed: TeamSlateFeed;
  detailBasePath: "/nba" | "/nfl";
}) {
  const cardPicks = feed.games.map(toEnrichedPick).filter(Boolean) as EnrichedPick[];

  return (
    <div className="space-y-4">
      {feed.gaps.length ? (
        <Card>
          <CardHeader>
            <CardTitle>Data Gaps</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {feed.gaps.map((gap) => (
              <Badge key={gap} variant="missing">
                {gap}
              </Badge>
            ))}
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>
            {feed.league} Slate ({feed.windowStart}
            {feed.windowEnd !== feed.windowStart ? ` → ${feed.windowEnd}` : ""})
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Matchup</TableHead>
                <TableHead>Time</TableHead>
                <TableHead>Book</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Edge</TableHead>
                <TableHead>Win%</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {feed.games.length ? (
                feed.games.map((game) => (
                  <TableRow key={game.gameId}>
                    <TableCell>
                      <Link
                        href={`${detailBasePath}/${game.gameId}`}
                        className="font-medium hover:underline"
                      >
                        {game.awayTeam} @ {game.homeTeam}
                      </Link>
                      {game.week != null ? (
                        <div className="text-xs text-muted-foreground">Week {game.week}</div>
                      ) : null}
                    </TableCell>
                    <TableCell>{formatDateTime(game.gameTimeUtc)}</TableCell>
                    <TableCell>{formatSpread(game.bookSpread)}</TableCell>
                    <TableCell>{formatSpread(game.modelSpread)}</TableCell>
                    <TableCell>{formatSpread(game.edgePts)}</TableCell>
                    <TableCell>{formatPct(game.homeWinProb)}</TableCell>
                    <TableCell>
                      <FreshnessBadge status={game.freshnessStatus} />
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={7} className="text-muted-foreground">
                    No games in serving window.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {cardPicks.length ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {cardPicks.map((pick) => (
            <Link key={pick.id} href={`${detailBasePath}/${pick.id}`}>
              <PickCard pick={pick} />
            </Link>
          ))}
        </div>
      ) : null}

      <p className="text-xs text-muted-foreground">
        {formatNumber(feed.games.length)} games · updated {formatDateTime(feed.generatedAt)}
      </p>
    </div>
  );
}
