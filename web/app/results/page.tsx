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
import { getResultsData } from "@/lib/data/results";
import { formatDate, formatNumber, formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function ResultsPage() {
  const data = await getResultsData();

  return (
    <div>
      <PageHeader
        title="Results"
        description="Live graded outcomes for model spread, winner, MLB home run, and PGA placement markets."
        meta={data.generatedAt}
      />

      {data.gaps.length ? (
        <div className="mb-4 flex flex-wrap gap-2">
          {data.gaps.map((gap) => (
            <Badge key={gap} variant="missing">
              {gap}
            </Badge>
          ))}
        </div>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Graded Success Rates</CardTitle>
        </CardHeader>
        <CardContent>
          <Table className="table-fixed">
            <TableHeader>
              <TableRow>
                <TableHead>League</TableHead>
                <TableHead>Market</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Sample</TableHead>
                <TableHead>W-L-P</TableHead>
                <TableHead>Hit Rate</TableHead>
                <TableHead>Flat ROI</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.summaries.map((row) => (
                <TableRow key={`${row.league}-${row.market}-${row.modelVersion}`}>
                  <TableCell className="font-medium">{row.league}</TableCell>
                  <TableCell>{row.market}</TableCell>
                  <TableCell>{row.modelVersion}</TableCell>
                  <TableCell>{formatNumber(row.sample)}</TableCell>
                  <TableCell>
                    {row.wins}-{row.losses}-{row.pushes}
                  </TableCell>
                  <TableCell>{formatPct(row.hitRate)}</TableCell>
                  <TableCell>{formatPct(row.roi)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Recent Game Grades</CardTitle>
          </CardHeader>
          <CardContent>
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
                {data.gameRows.slice(0, 12).map((row) => (
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
                      <Badge variant={row.spread_result === "win" ? "accent" : row.spread_result === "push" || !row.spread_result ? "outline" : "missing"}>
                        {row.spread_result ?? "n/a"}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={row.winner_result === "win" ? "accent" : row.winner_result ? "missing" : "outline"}>
                        {row.winner_result ?? "n/a"}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Player-Market Grades</CardTitle>
          </CardHeader>
          <CardContent>
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
                {data.mlbHrRows.slice(0, 8).map((row) => (
                  <TableRow key={`hr-${row.game_date}-${row.player_name}-${row.model_version}`}>
                    <TableCell>{formatDate(row.game_date)}</TableCell>
                    <TableCell>{row.player_name}</TableCell>
                    <TableCell>HR {row.top_k_bucket ?? "field"}</TableCell>
                    <TableCell>{row.model_version}</TableCell>
                    <TableCell>
                      <Badge variant={row.actual_home_run ? "accent" : row.actual_home_run === false ? "missing" : "outline"}>
                        {row.actual_home_run ? "hit" : row.actual_home_run === false ? "miss" : "n/a"}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
                {data.pgaRows.slice(0, 8).map((row) => (
                  <TableRow key={`pga-${row.event_key}-${row.player_name}-${row.model_version}`}>
                    <TableCell>{row.event_key}</TableCell>
                    <TableCell>{row.player_name}</TableCell>
                    <TableCell>Top 10</TableCell>
                    <TableCell>{row.model_version}</TableCell>
                    <TableCell>
                      <Badge variant={row.top10_hit ? "accent" : row.top10_hit === false ? "missing" : "outline"}>
                        {row.final_position ?? "n/a"}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
