import Link from "next/link";

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
import { isFiniteNumber } from "@/lib/data/json";
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

function modelWinner(game: TeamSlateGame) {
  const probability = game.homeWinProb;
  if (!isFiniteNumber(probability) || probability < 0 || probability > 1) return "n/a";
  if (probability === 0.5) return "Even · 50.0%";
  return probability > 0.5
    ? `${game.homeTeam} · ${formatPct(probability)}`
    : `${game.awayTeam} · ${formatPct(1 - probability)}`;
}

function FreshnessBadge({ status }: { status: FreshnessStatus }) {
  const labels = {
    fresh: "With book odds",
    stale: "Stale forecast",
    no_prediction: "Forecast missing",
    no_odds: "Model only",
  } satisfies Record<FreshnessStatus, string>;
  return <Badge variant={freshnessVariant(status)}>{labels[status]}</Badge>;
}

export function TeamSpreadBoard({
  feed,
  detailBasePath,
}: {
  feed: TeamSlateFeed;
  detailBasePath: "/markets/nba" | "/markets/nfl";
}) {
  const predictionTs = feed.games.map((game) => game.predictionTs)
    .filter((stamp): stamp is string => Boolean(stamp)).sort().at(-1);

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
            {feed.league} game forecasts ({feed.windowStart}
            {feed.windowEnd !== feed.windowStart ? ` → ${feed.windowEnd}` : ""})
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4 text-sm text-muted-foreground">
            Projected spread is from the home team&apos;s perspective. Model winner shows
            the team with the higher win probability. Book prices are required for edge or EV.
          </p>
          <Table className="min-w-[760px]">
            <TableHeader>
              <TableRow>
                <TableHead>Matchup</TableHead>
                <TableHead>Kickoff</TableHead>
                <TableHead>Projected home spread</TableHead>
                <TableHead>Model winner</TableHead>
                <TableHead>Book home spread</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {feed.games.length ? (
                feed.games.map((game) => {
                  const currentForecast = game.freshnessStatus === "fresh" || game.freshnessStatus === "no_odds";
                  return (
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
                      <TableCell>{currentForecast ? formatSpread(game.modelSpread) : "n/a"}</TableCell>
                      <TableCell>{currentForecast ? modelWinner(game) : "n/a"}</TableCell>
                      <TableCell>{formatSpread(game.bookSpread)}</TableCell>
                      <TableCell>
                        <FreshnessBadge status={game.freshnessStatus} />
                      </TableCell>
                    </TableRow>
                  );
                })
              ) : (
                <TableRow>
                  <TableCell colSpan={6} className="text-muted-foreground">
                    No games in serving window.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground">
        {formatNumber(feed.games.length)} games · forecasts updated {formatDateTime(predictionTs)}
      </p>
    </div>
  );
}
