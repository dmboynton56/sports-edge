import Link from "next/link";
import { PageHeader, SectionHeading } from "@/components/dashboard/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/dashboard/EmptyState";
import { Notice } from "@/components/dashboard/Notice";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { SportSwatch } from "@/components/dashboard/SportChip";
import { getPerformanceHistory } from "@/lib/data/performance";
import { getResultsData } from "@/lib/data/results";
import { isFiniteNumber } from "@/lib/data/json";
import { formatDate, formatNumber, formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function RecordPage() {
  const [history, results] = await Promise.all([
    getPerformanceHistory(),
    getResultsData(),
  ]);

  return (
    <div>
      <PageHeader
        title="Record"
        description="Backtest results, performance history, and official graded outcomes. Both live board health and historical ROI live here."
      />

      <SectionHeading title="Season performance" note="Backtest to date" />

      <Card className="overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Sport</TableHead>
              <TableHead className="hidden sm:table-cell">Model</TableHead>
              <TableHead>Market</TableHead>
              <TableHead className="text-right">Sample</TableHead>
              <TableHead className="text-right">ROI</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {history.records.map((record) => {
              const roi = record.roi;
              return (
                <TableRow key={`${record.sport}-${record.modelVersion}`}>
                  <TableCell><SportSwatch sport={record.sport} label={record.sport} /></TableCell>
                  <TableCell className="hidden sm:table-cell">{record.modelVersion}</TableCell>
                  <TableCell>{record.market}</TableCell>
                  <TableCell className="text-right">{formatNumber(record.sampleSize)}</TableCell>
                  <TableCell className={isFiniteNumber(roi) ? `figure text-right text-[17px] ${roi < 0 ? "text-destructive" : "text-positive"}` : "text-right text-sm"}>
                    {isFiniteNumber(roi) ? formatPct(roi) : "No odds history"}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Card>

      <SectionHeading title="Recent graded results" note="Live outcomes" />

      {results.gaps.length > 0 ? (
        <Notice
          className="mb-4"
          title={`${results.gaps.length} ${results.gaps.length === 1 ? "caveat" : "caveats"} on these results`}
          items={Array.from(new Set(results.gaps))}
        />
      ) : null}

      <div className="grid gap-3 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Recent game grades</CardTitle>
          </CardHeader>
          <CardContent>
            {results.gameRows.length === 0 ? (
              <EmptyState
                className="min-h-40 border-0 bg-transparent py-6"
                title="No games graded yet"
                description="Spread and winner outcomes appear here the morning after each slate."
              />
            ) : (
            <Table className="table-fixed">
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead>Game</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Spread</TableHead>
                  <TableHead>Winner</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.gameRows.slice(0, 8).map((row) => (
                  <TableRow key={`${row.league}-${row.game_date}-${row.away_team}-${row.home_team}-${row.model_version}`}>
                    <TableCell>{formatDate(row.game_date)}</TableCell>
                    <TableCell>
                      {row.away_team} @ {row.home_team}
                      <div className="text-xs text-muted-foreground">
                        {row.away_score}-{row.home_score}
                      </div>
                    </TableCell>
                    <TableCell>{row.model_version}</TableCell>
                    <TableCell>
                      <Badge variant={row.spread_result === "win" ? "positive" : row.spread_result === "push" || !row.spread_result ? "outline" : "missing"}>
                        {row.spread_result ?? "n/a"}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={row.winner_result === "win" ? "positive" : row.winner_result ? "missing" : "outline"}>
                        {row.winner_result ?? "n/a"}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent player-market grades</CardTitle>
          </CardHeader>
          <CardContent>
            {results.mlbHrRows.length === 0 && results.pgaRows.length === 0 ? (
              <EmptyState
                className="min-h-40 border-0 bg-transparent py-6"
                title="No player markets graded yet"
                description="Home-run and placement outcomes appear here once each event finishes."
              />
            ) : (
            <Table className="table-fixed">
              <TableHeader>
                <TableRow>
                  <TableHead>Date/Event</TableHead>
                  <TableHead>Player</TableHead>
                  <TableHead>Market</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Result</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.mlbHrRows.slice(0, 6).map((row) => (
                  <TableRow key={`hr-${row.game_date}-${row.player_name}-${row.model_version}`}>
                    <TableCell>{formatDate(row.game_date)}</TableCell>
                    <TableCell>{row.player_name}</TableCell>
                    <TableCell>HR {row.top_k_bucket ?? "field"}</TableCell>
                    <TableCell>{row.model_version}</TableCell>
                    <TableCell>
                      <Badge variant={row.actual_home_run ? "positive" : row.actual_home_run === false ? "missing" : "outline"}>
                        {row.actual_plate_appearances === 0
                          ? "void"
                          : row.actual_home_run
                            ? "hit"
                            : row.actual_home_run === false
                              ? "miss"
                              : "n/a"}
                      </Badge>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {row.american_price && row.odds_status ? "priced snapshot" : "model only"}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
                {results.pgaRows.slice(0, 6).map((row) => (
                  <TableRow key={`pga-${row.event_key}-${row.player_name}-${row.model_version}`}>
                    <TableCell>{row.event_key}</TableCell>
                    <TableCell>{row.player_name}</TableCell>
                    <TableCell>Top 10</TableCell>
                    <TableCell>{row.model_version}</TableCell>
                    <TableCell>
                      <Badge variant={row.top10_hit ? "positive" : row.top10_hit === false ? "missing" : "outline"}>
                        {row.final_position ?? "n/a"}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="mt-6 text-sm text-muted-foreground">
        <Link href="/data-quality" className="underline underline-offset-2 hover:text-foreground">
          Data quality
        </Link>
        {" · "}
        <Link href="/insights" className="underline underline-offset-2 hover:text-foreground">
          Insights
        </Link>
      </div>
    </div>
  );
}
